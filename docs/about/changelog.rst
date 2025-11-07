
ChangeLog
=========

All notable changes to SDialog will be documented here.

----

[0.3.3] 2025-10-30 🚀
---------------------

Added
^^^^^


* **sdialog.server**\ :

  * New module to serve agents via an Ollama/OpenAI-compatible REST API (works with UIs like Open WebUI) (#92)

* **sdialog**\ :

  * ``Dialog.from_huggingface()`` to load/download dialogues directly from Hugging Face datasets (#59)

Changed
^^^^^^^


* **sdialog.evaluation**\ :

  * LLM judge methods now accept additional user-defined template arguments (e.g., like ``document`` in `this example <https://sdialog.readthedocs.io/en/latest/examples/index.html#example-1-yes-no-relevance-judgment-with-reasoning>`_\ ) (#86)

* **sdialog.agents**\ :

  * Improved ``Agent`` initialization so agents can act as a proxy for external conversational systems (#90, fa1d8f3)

Fixed
^^^^^


* **sdialog.evaluation**\ :

  * Corrected Flesch Reading Ease and Gunning Fog score calculations (d1d4260)

----

[0.3.0] 2025-10-03 ✨
---------------------

Added
^^^^^


* **sdialog**\ : 

  * ``Context``\ : new class class to explicitly model the common/shared context of conversations (#73)
  * ``Dialog``\ : merge functionality - Added option to merge consecutive turns of the same speaker when loading a dialog (#77)
  * ``Dialog``\ : built-in string support - Added support to built-in str functions for ``Dialog`` class (#83)

* **sdialog.agents**\ : Added new ``sdialog.agents`` module and moved ``Agent`` class inside (#81)

  * ``Agent``\ : thinking capabilities - Agents can now handle internal thinking processes (#95)
  * ``Agent``\ : tools support - Added tools capabilities to Agents (e.g. RAG or any other function) (#84)

    * New tutorial for agents with tools and thoughts.

* **sdialog.generators**\ : 

  * ``ContextGenerator``\ : new class added to explicitly model the common/shared context of conversations (#73)
  * ``Paraphraser``\ : new class class to paraphrase dialogues (#76)

* **sdialog.evaluation**\ : 

  * ``LinguisticFeatureScore``\ : new class added to compute Flesch reading ease, Gunning fog, Hesitation rate, and/or Mean turn length (#63)

* **sdialog.personas**\ : 

  * ``Customer`` and ``SupportAgent``\ : new personas added for customer service dialogues (#85)
  * ``Persona``\ : Added static method to get the list of all attributes in ``Persona`` class (#79)

Changed
^^^^^^^


* **sdialog**\ : Improved metadata handling (#66)
* **sdialog.interpretability**\ : Improved and simplified the way inspection targets are defined in ``interpretability`` submodule (#78)
* **sdialog.evaluation.base**\ : 

  * ``LLMJudgeYesNoOutput``\ : Renamed attribute ``yes`` to ``positive`` (#86)
  * ``LLMJudgeScoreOutput``\ : Renamed attribute ``feedback`` to ``reason`` (#86)

Fixed
^^^^^


* **sdialog.generators**\ : Fixed potential bug in ``PersonaDialogGenerator`` class (#67)

Enhanced
^^^^^^^^


* **sdialog.agents**\ : Added ``base_model`` attribute to ``Agent`` to direclty access the LLM's underlying model for mechanistic interpretability (#74)
* **sdialog.config**\ : Added ``clear_cache()`` method to config (#75)

Documentation
^^^^^^^^^^^^^


* API Documentation: Refactored/cleaned all components and added docstrings with examples (#82, #88)
* Updated all tutorials to work with new code and added "Open in Colab" badges
* Completed API documentation for initial official release (#87)
* Automatic generation of ``llm.txt`` from API documentation (24f6ee6)

----

[0.1.0] 2025-08-05 🌱
---------------------

Added
^^^^^


* Multi-backend support (Hugging Face, Ollama, OpenAI, AWS)
* Enhanced persona generation (beyond initial ``PersonaDialogGenerator``\ )
* Interpretability module (\ ``sdialog.interpretability``\ ): inspectors, steerers, hooks, intruders
* Evaluation module (\ ``sdialog.evaluation``\ ): metrics, LLM-as-a-judge scoring, evaluators, dataset comparators

Changed
^^^^^^^


* Standardized / improved dialog format

Notes
^^^^^


* 
  ..

     500 commits since 0.0.2 (post-JSALT 2025 consolidation)


Pending
^^^^^^^


* Audio module (\ ``sdialog.audio``\ ) integration
* Documentation updates

----

[0.0.2] 2025-06-03 🔧
---------------------

Added
^^^^^


* ``language`` attribute to ``Persona`` class
* 
  ``PersonaDialogGenerator`` to ``generators`` module to support persona-based dialogue generatin with single LLM:

  .. code-block:: python

     from sdialog.generators import PersonaDialogGenerator

     dialog_generator = PersonaDialogGenerator(
         model=MODEL_NAME,
         persona_a=bob_persona,
         persona_b=alice_persona,
     )

     dialog_generator.generate().print()

Fixed
^^^^^


* Python 2 and 3 compatibility problem with scikit-learn (using version 0.20.1 from now on)
* PyPi: setup.py: ``long_description_content_type`` set to ``'text/markdown'``

----

[0.0.1] 2025-05-22 🎉
---------------------

*(initial release)*
