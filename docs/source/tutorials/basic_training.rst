Basic Training Tutorial
=======================

This tutorial will walk you through training your first agent with Meow Tea Taro.

Prerequisites
-------------

Before starting, make sure you have:

* Python 3.10 or higher installed
* The meow_tea_train package installed
* Access to TextWorld or ALFWorld environments

Step 1: Prepare Your Data
--------------------------

First, we need to generate training data from the environment:

.. code-block:: bash

   python -m meow_tea_experiments.data_generation.generate_multiturn_data \
       --env_name textworld \
       --instance_dir ./data/instances \
       --instance_id_range 0 799 \
       --task_prefix w2-o3-q4 \
       --out_dir ./data/processed \
       --train_type ppo \
       --split train

This will create a ``train.jsonl`` file in your output directory.

Step 2: Configure Training
---------------------------

Create a configuration file for your training run:

.. code-block:: python
   :caption: config.py
   :linenos:

   training_config = {
       "model_name": "gpt2",
       "learning_rate": 1e-4,
       "batch_size": 32,
       "num_epochs": 10,
       "gradient_accumulation_steps": 4,
   }

.. note::
   
   Adjust these parameters based on your GPU memory and dataset size.

Step 3: Train the Model
------------------------

Now we can start training:

.. code-block:: python
   :emphasize-lines: 5,6

   from meow_tea_train import Trainer
   from config import training_config
   
   trainer = Trainer(**training_config)
   trainer.load_data("./data/processed/train.jsonl")
   trainer.train()

Training Progress
^^^^^^^^^^^^^^^^^

You should see output like this::

   Epoch 1/10: 100%|████████████| 1000/1000 [05:32<00:00, 3.01it/s]
   Loss: 2.453, Accuracy: 0.723
   
   Epoch 2/10: 100%|████████████| 1000/1000 [05:31<00:00, 3.02it/s]
   Loss: 1.892, Accuracy: 0.812

.. warning::
   
   If you see NaN losses, reduce your learning rate or gradient accumulation steps.

Step 4: Evaluate
----------------

After training, evaluate your model:

.. code-block:: python

   results = trainer.evaluate("./data/processed/test.jsonl")
   print(f"Test Accuracy: {results['accuracy']:.3f}")

Next Steps
----------

Now that you've trained your first model, you can:

* :doc:`custom_environments` - Add support for new environments
* :doc:`../api/modules` - Explore the full API
* Try different training strategies (SFT vs PPO)

.. seealso::
   
   * :ref:`Advanced Training Options <advanced-training>`
   * `Paper: Multi-turn Agentic RL <https://arxiv.org/...>`_