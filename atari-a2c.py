#!/usr/bin/env python3
# coding: utf-8

import math
import numpy as np
from tqdm import tqdm

import gym
from baselines.common.atari_wrappers import NoopResetEnv,EpisodicLifeEnv,MaxAndSkipEnv,WarpFrame
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import keras
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
import keras.backend as K
from keras.optimizers import RMSprop

def a2c_model(obs_shape, action_num):
    input_layer = Input(shape=list(obs_shape) )
    
    conv_orth = keras.initializers.Orthogonal(gain=math.sqrt(2.0), seed=None)
    layer = Conv2D(32, 8, strides=4, activation='relu', init=conv_orth)(input_layer)
    layer = Conv2D(64, 4, strides=2, activation='relu', init=conv_orth)(layer)
    layer = Conv2D(64, 3, strides=1, activation='relu', init=conv_orth)(layer)
    layer = Flatten()(layer)
    
    layer = Dense(512, activation='relu', init=conv_orth)(layer)

    critic_orth = keras.initializers.Orthogonal(gain=1.0, seed=None)
    critic = Dense(1, init=critic_orth)(layer)

    actor_orth = keras.initializers.Orthogonal(gain=0.01, seed=None)
    actor = Dense(action_num, activation='softmax', init=actor_orth)(layer)

    model = Model(inputs=input_layer, outputs=[critic, actor])
    return model

def a2c_update(model, num_steps, num_procs):
    def get_updates(model, actions, drewards, num_steps, num_procs):
        VALUE_LOSS_COEF = 0.5
        ENTROPY_COEF = 0.01
        MAX_GRAD_NORM = 0.5
        LR = 7e-4
        EPS = 1e-5
        ALPHA = 0.99

        # log_probs: (80,4)
        pred_values, action_probs = model.output
        log_probs = K.log(action_probs + 1e-5)

        # action_log_probs: (80, 1)
        # a[i][j] = log_probs[i][actions[i][j]]
        pred_actions = K.map_fn(lambda x: K.gather(x[0], x[1]), 
            [log_probs, K.reshape(actions, [-1])], dtype="float32")
    
        # reshape variables.
        pred_values = K.reshape(pred_values, [-1, 1])
        pred_actions = K.reshape(pred_actions, [-1, 1])
        drewards = K.reshape(drewards, [-1, 1]) 

        # create critic loss func. 
        advantages = drewards - pred_values
        value_loss = K.mean(K.pow(advantages, 2))

        # create actor loss func. 
        action_gain = K.mean(pred_actions * K.stop_gradient(advantages))
        dist_entropy = -K.mean(K.sum(log_probs * action_probs, axis=-1))

        # create total_loss.
        total_loss = (value_loss * VALUE_LOSS_COEF - action_gain - dist_entropy * ENTROPY_COEF)
                      
        return RMSprop(lr=LR, epsilon=EPS, rho=ALPHA, clipnorm=MAX_GRAD_NORM).get_updates(
                params=model.trainable_weights, loss=total_loss) 

    actions = K.placeholder(shape=(None, 1), dtype="int32")
    drewards = K.placeholder(shape=(None, 1), dtype="float32")

    update_func = K.function(
        [model.input, actions, drewards], [], 
         updates=get_updates(model, actions, drewards, num_steps, num_procs)
    )
    return update_func

class Environment:
    NUM_STACK_FRAME = 4
    TOTAL_FRAMES=10e+6
    NUM_ADVANCED_STEP = 5
    NUM_TRAIN_PROCS = 16
    NUM_UPDATES = int(TOTAL_FRAMES / NUM_ADVANCED_STEP / NUM_TRAIN_PROCS)
    GAMMA = 0.99

    def __init__(self, env_name, train=True, model_path=""):
        self.train = train
        if self.train:
            self.env = self.make_env(env_name, self.NUM_TRAIN_PROCS)
            self.num_procs = self.NUM_TRAIN_PROCS
        else: # play
            self.env = self.make_env(env_name, 1)
            self.num_procs = 1

        # action_num: difference
        self.action_num = self.env.action_space.n
        # env_obs_shape: (84, 84, 1)
        # obs_shape: (84, 84, 4)
        env_obs_shape = self.env.observation_space.shape
        self.obs_shape = (*env_obs_shape[:2], env_obs_shape[-1] * self.NUM_STACK_FRAME)

        # model 
        # input: (None, 84, 84, 4)
        # output: [(1,),(n_out,)]
        if self.train:
            self.model = a2c_model(self.obs_shape, self.action_num)
            self.update = a2c_update(self.model, self.NUM_ADVANCED_STEP, self.NUM_TRAIN_PROCS)
        else:
            self.model = keras.models.load_model(model_path)

        # compile model. 
        self.model.summary()


    def make_env(self, env_name, num_procs):
        def atari_env(env_name, seed):
            def env_func():
                env = gym.make(env_name)
                env = NoopResetEnv(env, noop_max=30)
                env = MaxAndSkipEnv(env, skip=4)
                env.seed(seed)

                env = EpisodicLifeEnv(env)
                env = WarpFrame(env)
                return env
            return env_func
        
        envs = [atari_env(env_name, i+1) for i in range(num_procs)]
        env = SubprocVecEnv(envs)
        return env

    def run(self):
        # current_obs: (16, 84, 84, 4)
        current_obs = np.zeros([self.num_procs, *self.obs_shape])
        # episodic_rewards: (16, 1)
        episode_rewards = np.zeros([self.num_procs, 1])
        # final_rewards: (16, 1)
        final_rewards = np.zeros([self.num_procs, 1])

        # obs: (16, 84, 84, 1)
        obs = self.env.reset()
        current_obs[:, :, :, :1] = obs  

        steps_obs      = np.zeros([self.NUM_ADVANCED_STEP + 1, self.num_procs, *self.obs_shape])
        steps_masks    = np.zeros([self.NUM_ADVANCED_STEP + 1, self.num_procs, 1])
        steps_rewards  = np.zeros([self.NUM_ADVANCED_STEP, self.num_procs, 1])
        steps_actions  = np.zeros([self.NUM_ADVANCED_STEP, self.num_procs, 1], dtype=np.int32)
        steps_drewards = np.zeros([self.NUM_ADVANCED_STEP + 1, self.num_procs, 1])
        steps_obs[0] = current_obs

        for idx in tqdm(range(self.NUM_UPDATES)):
            for step in range(self.NUM_ADVANCED_STEP):
                _, action_probs = self.model.predict(steps_obs[step] / 255.)
                action_probs += 1e-5
                action_probs /= action_probs.sum(axis=-1).reshape(-1,1)
                action = np.argmax(np.array([np.random.multinomial(1, x) for x in action_probs]), axis=1)
                obs, reward, done, _ = self.env.step(action)

                # action: (16,) -> (16, 1)
                action = action.reshape(-1,1)
                # obs: (16, 84, 84, 1)
                # reward: (16,) -> (16, 1)
                reward = reward.reshape(-1,1)
                # done: (16,) -> mask: (16, 1)
                mask = 1.0 - done.reshape(-1,1)

                # finalize done process
                episode_rewards += reward
                final_rewards[done] = episode_rewards[done]
                episode_rewards[done] = 0.0
                current_obs[done] = 0.0

                # stack obs
                current_obs[:,:,:,1:] = current_obs[:,:,:,:-1]
                current_obs[:,:,:,:1] = obs

                # update step variables.
                steps_obs[step+1]   = current_obs
                steps_masks[step+1] = mask
                steps_rewards[step] = reward
                steps_actions[step] = action

            if self.train:
                # pred_value: (16,) -> (16, 1)
                stepend_obs = steps_obs[-1]  / 255.
                pred_value, _ = self.model.predict(stepend_obs)
                pred_value = pred_value.reshape(-1,1)

                # calculate discounted values.
                steps_drewards[-1] = pred_value
                for step in reversed(range(self.NUM_ADVANCED_STEP)):
                    steps_drewards[step] = steps_rewards[step] + self.GAMMA * steps_drewards[step + 1] * steps_masks[step + 1]

                # update network.
                flatten_obs      = steps_obs[:-1].reshape(-1, *self.obs_shape) / 255.
                flatten_actions  = steps_actions.reshape(-1, 1)
                flatten_drewards = steps_drewards[:-1].reshape(-1, 1)
                self.update([flatten_obs, flatten_actions, flatten_drewards])
            
                steps_obs[0] = steps_obs[-1]
                steps_masks[0] = steps_masks[-1]

                # output train logs.
                if idx % 100 == 0:
                    print("finished frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
                          format(idx*self.num_procs*self.NUM_ADVANCED_STEP,
                                 final_rewards.mean(), np.median(final_rewards),
                                 final_rewards.min(),final_rewards.max()))

                # save models
                if idx % 12500 == 0:
                    self.model.save('weight_'+str(idx)+'.pth')
            else: # play
                self.env.render()
        
        if self.train:
            # save final model
            self.model.save('weight_end.pth')


if __name__=="__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: {} env_id is_train model_path".format(sys.argv[0]))
        print()
        print("Atari Environments List")
        atari_specs = [spec for spec in gym.envs.registry.all() 
                if spec.id.endswith("NoFrameskip-v4") and not spec.id.endswith("ramNoFrameskip-v4")]
        for idx, spec in enumerate(atari_specs):
            print("No.{:03}: {}".format(idx, spec.id) )
        exit()

    env_id = sys.argv[1]
    mode = sys.argv[2]
    if mode=="train":
        breakout_env = Environment(env_id, True)
    else: # play
        model_path = sys.argv[3]
        breakout_env = Environment(env_id, False, model_path)
    
    breakout_env.run()
