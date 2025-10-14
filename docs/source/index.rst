.. meow-tea-taro documentation master file, created by
   sphinx-quickstart on Mon Oct 13 05:42:19 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

meow-tea-taro documentation
===========================

**Meow Tea Taro** is a Multi-turn Agentic Reinforcement Learning Framework for training agents in interactive environments like TextWorld and ALFWorld.

.. note::
   This project is under active development. APIs may change between versions.

We use verl (add url: https://github.com/volcengine/verl) as the backbone architecture. 
We make minimum modifications to **verl** and provide a **ready-to-use** and **customizable** framework for training multi-turn agentic RL.
The learning curve for **meow-tea-taro** would be very low if you are already familiar with **verl**.

Features
--------

* Multi-turn interaction support for complex environments
* Integration with TextWorld and ALFWorld
* PPO and SFT training strategies
* Modular architecture for easy extension

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/quick_install
   getting_started/quick_start_textworld

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_training
   tutorials/custom_environments
   tutorials/data_preparation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Example
-------------

Here's a simple example to get you started:

.. code-block:: python

   from meow_tea_train import TrainingAgent
   from meow_tea_experiments.data_generation import generate_multiturn_data
   
   # Generate training data
   data = generate_multiturn_data(
       env_name="textworld",
       instance_dir="./instances",
       instance_start=0,
       instance_end=100,
       task_prefix="w2-o3-q4",
       train_type="ppo"
   )
   
   # Train an agent
   agent = TrainingAgent()
   agent.train(data)
