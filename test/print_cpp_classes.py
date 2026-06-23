#!/bin/env python

import inspect

import stablebear._sb_cpp as cpp

classes = inspect.getmembers(cpp, inspect.isclass)

for name, _ in classes:
    print(f"Class: {name}")
