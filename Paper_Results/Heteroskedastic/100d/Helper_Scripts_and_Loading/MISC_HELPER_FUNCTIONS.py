#!/usr/bin/env python
# coding: utf-8

# # Miscellaneous Helper Function(s)

# This function ensures that the minimum height of [Kidger and Lyons (2020) COLT](http://proceedings.mlr.press/v125/kidger20a.html) is achieved.

# In[ ]:


def minimum_height_updater(list_input):
    for i_loop in range(len(list_input)):
        # Get List Item
        current_list_loop_item = list_input[i_loop]
        # Apply 
        list_input[i_loop] = int(max(current_list_loop_item,problem_dim+output_dim+2))
    # Return Minimum Width Network
    return list_input

