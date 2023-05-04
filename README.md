# Visual Policy Chaining

This project is based on the original PyTorch implementation of [Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization](https://clvrai.com/skill-chaining), published in CoRL 2021. Here, we try to modify the source code to do the following:

- Generate a visual control policy, i.e., a control policy directly from raw images instead of extracted states.
- Train offline-RL algorithms such as BC, DDPG, SAC, and TD3+BC using expert demonstrations which were obtained from a trained state-based agent.



### Run 

Instructions to run this project can be found in [skill-chaining README](./README_skill-chaining.md).