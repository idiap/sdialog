# EACL 2026 - System Demonstration Video

The demo illustrates the key capabilities of th SDialog toolkit through a practical examples following a simple call-center use case, showing how SDialog enables researchers to build, benchmark, and analyze conversational AI systems systematically, from initial agent design through dialog generation to comprehensive evaluation.

**(Click the image below to watch the demonstration video)**

[![SDialog System Demo Video](https://img.youtube.com/vi/oG_jJuU255I/0.jpg)](https://www.youtube.com/watch?v=oG_jJuU255I)

Alternatively, link to the video: https://www.youtube.com/watch?v=oG_jJuU255I

## Replicating the Demo

The files shown in the video demo can be found in the [`tutorials/demo`](tutorials/demo) folder, where you can see all the details and replicate the results. Note that this folder contains a slightly more complex version than shown in the video: the agent has three tools instead of two (one to verify the user, one to change the billing address that requires verification, and one to get the service plans that does not require verification). As such, dialog generation simulates two types of customers: one that requires the agent to verify the user to change the address, and another that does not (where the customer only asks about service plans). The evaluation also evaluates these two types of customers separately.
