# 🐈 🍵 Meow-Tea-Taro 💜: A Modular Multi-turn Agentic RL Framework with Configurable Environments 🌎, Policies 🤖, and Rewards ⭐


## Table of Contents
📖 [Overview](#overview)  
🚀 [Quick Start](#quick-start)  
🔔 [News](#news)  
📈 [Recipes](#recipes)  
🔑 [Key Findings](#key-findings)  
🧋 [Build Your Own Agentic Pipelines!](#build-your-own-agentic-pipelines)  
📚 [Citation](#citation)  

## 📖 Overview
**Welcome to the Meow-Tea Café! ☕️**

Just as "Multi" sounds like "Meow-tea" 🐈 🍵, our RL framework brings together the best ingredients for cooking up powerful Multi-turn Agentic RL solutions. 

### What's on the Menu?
At Meow-Tea Café, we serve a diverse menu of **agentic dishes** (tasks) ranging from text-based adventures to real-world software engineering challenges. Each dish in our [`meow_tea_train/agentic_menu/`](./meow_tea_train/agentic_menu/) represents a different agentic task:

- 🎮 **TextWorld**: Navigate complex text-based environments
- 🏠 **ALFWorld**: Master situated household tasks
- 💻 **SWE-Gym**: Tackle real software engineering problems
- *...and more specialty dishes coming soon!*


### The Art of RL Cooking 👨‍🍳
For each dish on our menu, we identify three essential **RL cooking processes** that can bring out its best flavors. We've made these components **fully configurable** in our framework:

1. **🌎 Environments** - The foundation of your dish (the agentic task itself, sync vs async rollout, tool use, thinking abilities)
2. **🤖 Policies** - Your RL cooking technique (PPO, GRPO, RLOO, and more)
3. **⭐ Rewards** - The perfect heat control and timing (single vs dense rewards, verified vs learned rewards)

These three pillars of RL cooking are the heart of our framework, allowing **RL cooking lovers** - whether you're a researcher, practitioner, or student - to experiment and explore innovative ways to think about and solve agentic tasks and challenges.

Can't find your favorite dish on our menu? No problem! The Meow-Tea Café includes a special [**`build_your_own`**](./meow_tea_train/agentic_menu/build_your_own/) section, where we'll walk you through creating the agentic task (dish) you want to cook. 


### Our Recipes and Research Insights
We provide tested recipes for different agentic tasks. These recipes are training configurations that have been validated through our experiments. You can find them under [`examples/{agentic_task}`](./examples/). 

Our paper presents systematic findings on **what works and what doesn't** for multi-turn agentic RL. Key research questions we address include:

- *Can we train agents on simpler environments and expect them to perform well on complex ones?*
- *How do different RL algorithms impact multi-turn RL training?*
- *Is there an optimal ratio of SFT:RL data given fixed budget?*
- *How does the density of rewards impact multi-turn RL training?*
- *...and more in our paper*


## 🚀 Quick Start

Please refer to [the documentation](./docs/source/getting_started/quick_install.rst) for a quick docker setup. 

Clone latest meow-tea-taro repository and install:
```bash
git clone git@github.com:ruiyiw/meow-tea-taro.git
cd meow-tea-taro
pip install -e .
```
That's it!

Run a quick example of multi-turn PPO on TextWorld tasks: 
```bash
sh examples/textworld/run_ppo_qwen-1.5b.sh
```

## 🔔 News
- 🎉 **[10/21/2025]** Meow-Tea-Taro codebase is now open-source! Recipes, datasets, and model checkpoints available.
- 🎉 **[10/01/2025]** Paper "A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning" released

## 📈 Recipes


## 🔑 Key Findings

## 🧋 Build Your Own Agentic Pipelines!

## 📚 Citation
```bibtex
@misc{wang2025practitionersguidemultiturnagentic,
      title={A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning}, 
      author={Ruiyi Wang and Prithviraj Ammanabrolu},
      year={2025},
      eprint={2510.01132},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.01132}, 
}
```

---

**Join our community of RL chefs! 👨‍🍳👩‍🍳** At Meow-Tea Café, we're passionate about **promoting open-source RL recipes and models**. We welcome contributions, new recipes, and fresh ideas to make Meow-Tea Café even better. We're committed to continuously updating our café. 

**Stay tuned as we keep expanding our menu and refining our recipes!**