import os
from typing import Any
import cv2

import numpy as np
from termcolor import colored
import gym
import d3rlpy

from tqdm import tqdm
from datetime import datetime

import argparse
from robot_learning.environments import make_env
import imageio


class ImageObsWrapper(gym.Wrapper):
    def __init__(self, env, height=500, width=500):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=[3, height, width], dtype=np.uint8
                )
        self.action_space = self.action_space['default']
        self.height = height
        self.width = width

    def reset(self):
        ob = self.env.reset()
        return self._get_obs(ob)

    def step(self, ac):
        ac = {'default': ac}
        ob, reward, done, info = self.env.step(ac)
        return self._get_obs(ob), reward, done, info

    def _get_obs(self, ob):
        # ob = self.render("rgb_array")[0]
        ob = ob['camera_ob'][0]
        ob = cv2.resize(ob, dsize=(self.height, self.width), interpolation=cv2.INTER_CUBIC)
        return ob.transpose(2, 0, 1)


class BC:
    def __init__(self, agent_type, mdp, ckpt_file_path=None):
        if agent_type == 'BC':
            self.agent = d3rlpy.algos.BC(use_gpu=False, scaler='pixel')
        elif agent_type == 'TD3+BC':
            self.agent = d3rlpy.algos.TD3PlusBC(use_gpu=False, scaler='pixel')
        elif agent_type == 'DDPG':
            self.agent = d3rlpy.algos.DDPG(use_gpu=False, scaler='pixel')
        elif agent_type == 'SAC':
            self.agent = d3rlpy.algos.SAC(use_gpu=False, scaler='pixel')
        elif agent_type == 'AWAC':
            self.agent = d3rlpy.algos.AWAC(use_gpu=False, scaler='pixel')
        else:
            raise Exception("Unknown Agent Type!!")
        if isinstance(mdp, d3rlpy.dataset.MDPDataset):
            self.agent.build_with_dataset(mdp)
        elif isinstance(mdp, gym.core.Env):
            self.agent.build_with_env(mdp)
        else:
            raise Exception("mdp object should be either an instance of \"d3rlpy.dataset.MDPDataset\" or \"gym.core.Env\".")
        if ckpt_file_path:
            self.agent.load_model(ckpt_file_path)
    

    def __call__(self, state):
        return self.agent.predict([state])[0]


class Evaluation:
    def __init__(self, args):
        self.args = args

        # default arguments from skill-chaining
        from policy_sequencing_config import create_skill_chaining_parser
        parser = create_skill_chaining_parser()
        self._config, unparsed = parser.parse_known_args()

        # set environment specific parameters
        setattr(self._config, 'algo', args.algo)
        setattr(self._config, 'furniture_name', args.furniture_name)
        setattr(self._config, 'env', 'IKEASawyerDense-v0')
        setattr(self._config, 'robot_ob', False)
        setattr(self._config, 'object_ob', False)
        setattr(self._config, 'object_ob_all', False)
        setattr(self._config, 'visual_ob', True)
        setattr(self._config, 'encoder_type', 'cnn')
        setattr(self._config, 'subtask_ob', True)
        setattr(self._config, 'depth_ob', False)
        setattr(self._config, 'segmentation_ob', False)
        setattr(self._config, 'phase_ob', False)
        setattr(self._config, 'screen_width', 500)
        setattr(self._config, 'screen_height', 500)
        setattr(self._config, 'demo_path', args.demo_path)
        setattr(self._config, 'num_connects', args.num_connects)
        setattr(self._config, 'run_prefix', args.run_prefix)
        setattr(self._config, 'is_train', False)
        setattr(self._config, 'record_video', False)
        if args.preassembled >= 0:
            preassembled = [i for i in range(0, args.preassembled+1)]
            setattr(self._config, 'preassembled', preassembled)
        
        # prepare video directory
        self.p2_ckpt_name = args.p2_checkpoint.split('/')[-1].split('.')[0]
        self.videos_dir = '/'.join(os.path.join(args.logdir, args.p2_checkpoint).split('/')[:-1]) + '/videos'
        if not os.path.exists(self.videos_dir):
            os.makedirs(self.videos_dir)

        env_kwargs = self._config.__dict__.copy()
        self._env = ImageObsWrapper(gym.make(self._config.env, **env_kwargs), self.args.obs_height, self.args.obs_width)
        # self._env = make_env(self._config.env, self._config)


    # def get_ob_image(self, env):
    #     ob_image = env.render("rgb_array")
    #     if len(ob_image.shape) == 4:
    #         ob_image = ob_image[0]
    #     if np.max(ob_image) <= 1.0:
    #         ob_image *= 255.0
    #     ob_image = ob_image.astype(np.uint8)

    #     # # DEBUG observation image
    #     # sample_img = ob_image
    #     # sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR)
    #     # cv2.imwrite('./input_image_bc_eval_test_p2.png', sample_img)
    #     # import pdb; pdb.set_trace()

    #     ob_image = cv2.resize(ob_image, (self.args.env_image_size, self.args.env_image_size))
    #     ob_image = np.transpose(ob_image, (2, 0, 1))
    #     ob_image = torch.from_numpy(ob_image).double().cuda()
    #     if self.transform:
    #         ob_image = self.transform(ob_image)

    #     return ob_image[None, :, :, :]

    def _save_video(self, fname, frames, fps=15.0):
        """ Saves @frames into a video with file name @fname. """
        path = os.path.join(self.videos_dir, fname)

        if np.issubdtype(frames[0].dtype, np.floating):
            for i in range(len(frames)):
                frames[i] = frames[i].astype(np.uint8)
        imageio.mimsave(path, frames, fps=fps)
        print(f'Video saved: {path}')
        return path

    def _store_frame(self, env, ep_len, ep_rew, info={}):
        """ Renders a frame. """
        color = (200, 200, 200)

        # render video frame
        frame = env.render("rgb_array")
        if len(frame.shape) == 4:
            frame = frame[0]
        if np.max(frame) <= 1.0:
            frame *= 255.0
        frame = frame.astype(np.uint8)

        h, w = frame.shape[:2]
        if h < 512:
            h, w = 512, 512
            frame = cv2.resize(frame, (h, w))
        frame = np.concatenate([frame, np.zeros((h, w, 3))], 0)
        scale = h / 512

        # add caption to video frame
        if self._config.record_video_caption:
            text = "{:4} {}".format(ep_len, ep_rew)
            font_size = 0.4 * scale
            thickness = 1
            offset = int(12 * scale)
            x, y = int(5 * scale), h + int(10 * scale)
            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
            for i, k in enumerate(info.keys()):
                v = info[k]
                key_text = "{}: ".format(k)
                (key_width, _), _ = cv2.getTextSize(
                    key_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness
                )

                cv2.putText(
                    frame,
                    key_text,
                    (x, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (66, 133, 244),
                    thickness,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    str(v),
                    (x + key_width, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return frame


    def simulate(self, policy_ckpt):
        agent = BC(self.args.agent_algo, self._env, policy_ckpt)

        subtask = 0
        done = False
        ep_len = 0
        ep_rew = 0
        ob_next = self._env.reset()

        record_frames = []
        record_frames.append(self._store_frame(self._env, ep_len, ep_rew))
        
        while not done:
            action = agent(np.array(ob_next, dtype=np.uint8))
            if len(action.shape) == 2:
                action = action[0]

            ob_next, reward, done, info = self._env.step(action)
            ep_len += 1
            ep_rew += reward

            if "subtask" in info and subtask != info["subtask"]:
                print(colored(f'Completed subtask {subtask}', "yellow"))
                subtask = info["subtask"]
                done = True

            # terminal/goal condition for policy sequencing algorithm (since we're only training 2 sub-policies)
            if self._config.algo == 'ps' and info['subtask'] == 2:
                done = True
                info['episode_success'] = True

            frame_info = info.copy()
            record_frames.append(self._store_frame(self._env, ep_len, ep_rew, frame_info))

        print(colored(f"Episode Total Rewards: {ep_rew}, Episode Length: {ep_len}", "yellow"))
        if 'episode_success' in info and info['episode_success']:
            print(colored(f"{info['episode_success']}!", "yellow"), "\n")
            total_success += 1
            num_subtask2_success += 1
            num_subtask1_success += 1 # since sub-task 2 is completed, sub-task 1 must have been completed
        else:
            if info["subtask"] == 1:
                num_subtask1_success += 1
        
        s_flag = 's' if 'episode_success' in info and info['episode_success'] else 'f'
        self._save_video(f'{self.p2_ckpt_name}_simulation_{ep_rew}_{subtask}_{s_flag}.mp4', record_frames)


    def evaluate(self, policy1_ckpt, policy2_ckpt):
        policies = []
        policies.append(BC(self._env, policy1_ckpt))
        policies.append(BC(self._env, policy2_ckpt))

        num_subtask2_success = 0
        num_subtask1_success = 0
        total_success, total_rewards, total_lengths, total_subtasks = 0, 0, 0, 0
        for ep in range(self.args.num_eval_eps):
            ob_next = self._env.reset()

            done = False
            ep_len = 0
            ep_rew = 0

            record_frames = []
            if self.args.is_eval:
                record_frames.append(self._store_frame(self._env, ep_len, ep_rew))

            subtask = 0
            while not done:
                action = policies[subtask](ob_next)
                if len(action.shape) == 2:
                    action = action[0]

                ob_next, reward, done, info = self._env.step(action.detach().cpu().numpy())
                ep_len += 1
                ep_rew += reward

                if "subtask" in info and subtask != info["subtask"]:
                    print(colored(f'Completed subtask {subtask}', "yellow"))
                    subtask = info["subtask"]

                # terminal/goal condition for policy sequencing algorithm (since we're only training 2 sub-policies)
                if self._config.algo == 'ps' and info['subtask'] == 2:
                    done = True
                    info['episode_success'] = True

                if self.args.is_eval:
                    frame_info = info.copy()
                    record_frames.append(self._store_frame(self._env, ep_len, ep_rew, frame_info))

            print(colored(f"Current Episode Total Rewards: {ep_rew}, Episode Length: {ep_len}", "yellow"))
            if 'episode_success' in info and info['episode_success']:
                print(colored(f"{info['episode_success']}!", "yellow"), "\n")
                total_success += 1
                num_subtask2_success += 1
                num_subtask1_success += 1 # since sub-task 2 is completed, sub-task 1 must have been completed
            else:
                if info["subtask"] == 1:
                    num_subtask1_success += 1
            if self.args.is_eval:
                s_flag = 's' if 'episode_success' in info and info['episode_success'] else 'f'
                self._save_video(f'{self.p2_ckpt_name}_ep_{ep}_{ep_rew}_{subtask}_{s_flag}.mp4', record_frames)
            total_rewards += ep_rew
            total_lengths += ep_len
            total_subtasks += subtask
        # output success rate of sub-tasks 1 and 2
        print(f'Success rate of Sub-task 2: {(num_subtask2_success / self.args.num_eval_eps) * 100}%')
        print(f'Success rate of Sub-task 1: {(num_subtask1_success / self.args.num_eval_eps) * 100}%')
        return total_success, total_rewards, total_lengths, total_subtasks


def main():
    parser = argparse.ArgumentParser()

    ## training
    parser.add_argument('--start_epoch', type=int, default=0, help="starting epoch for training")
    parser.add_argument('--end_epoch', type=int, default=1000, help="ending epoch for training")
    parser.add_argument('--seed', type=int, default=1234, help="torch seed value")
    parser.add_argument('--num_threads', type=int, default=1, help="number of threads for execution")
    parser.add_argument('--subtask_id', type=int, default=-1, help="subtask_id is used to retrieve subtask_id data from demonstrations")
    parser.add_argument('--p1_checkpoint', type=str, default='epoch_12.pth', help="policy 1 checkpoint file (frozen but use during training's validation)")

    ## logs
    parser.add_argument('--run_name_postfix', type=str, default=None, help="run_name_postfix")
    parser.add_argument('--wandb', type=bool, default=False, help="learning curves logged on weights and biases")
    parser.add_argument('--logdir', type=str, default='logs', help="Directory to store logs in")

    ## validation arguments
    parser.add_argument('--num_eval_eps', type=int, default=20, help="number of episodes to run during evaluation")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluation_interval")

    ## evaluation arguments
    parser.add_argument('--p2_checkpoint', type=str, default='epoch_12.pth', help="policy 2 checkpoint file")

    ## skill-chaining args
    parser.add_argument('--furniture_name', type=str, default='chair_ingolf_0650', help="furniture_name")
    parser.add_argument('--demo_path', type=str, default='demos/chair_ingolf/Sawyer_chair_ingolf_0650_0', help="demo_path")
    parser.add_argument('--run_prefix', type=str, default='p0', help="run_prefix")
    parser.add_argument('--algo', type=str, default='gail', help="algo")
    parser.add_argument('--preassembled', type=int, default=-1, help="preassembled")
    parser.add_argument('--num_connects', type=int, default=1, help="num_connects")

    parser.add_argument('--agent_algo', type=str, default="BC", choices=['BC', 'TD3+BC', 'DDPG', 'SAC', 'AWAC'], help="Algorithm to use for the learning agent")
    parser.add_argument('--obs_height', type=int, default=100, help="Height of the observation to use for agent training")
    parser.add_argument('--obs_width', type=int, default=100, help="Width of the observation to use for agent training")

    args = parser.parse_args()

    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.set_num_threads(args.num_threads)

    # torch.set_num_threads(1)
    # torch.multiprocessing.set_sharing_strategy('file_system') # for RuntimeError: Too many open files. Communication with the workers is no longer possible.
    # device = torch.device("cuda")

    if args.run_name_postfix:
        run_name = f'BC_{datetime.now().strftime("%m.%d.%H.%M.%S")}_{args.run_name_postfix}'
    else:
        run_name = f'BC_{datetime.now().strftime("%m.%d.%H.%M.%S")}'

    # load from checkpoint
    

    evaluation_obj = Evaluation(args)

    evaluation_obj.simulate(args.p1_checkpoint)
    # total_success, total_rewards, total_lengths, total_subtasks = evaluation_obj.evaluate(args.p1_checkpoint)
    # print(f'Success rate: {(total_success / args.num_eval_eps) * 100}%')
    # print(f'Average rewards: {total_rewards / args.num_eval_eps}')
    # print(f'Average episode length: {total_lengths / args.num_eval_eps}')
    # print(f'Average subtasks: {total_subtasks / args.num_eval_eps}')


if __name__ == '__main__':
    main()


