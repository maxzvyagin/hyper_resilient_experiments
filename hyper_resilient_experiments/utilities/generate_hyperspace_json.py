#!/usr/bin/env python

"""Utility script for interactively generating .json file of bounds to plug into HyperSpace for HyperResilient"""

import json
import sys


def create_json():
    print("Now generating HyperSpace bounds json file for Hyper Resilient optimization search...")
    output = input("Enter name of output file: \n")
    param_dict = {}
    while True:
        print("Enter q to quit and write the file.")
        param = input("Enter name of the parameter, no spaces: ")
        param = param.strip()
        if param == "q" or param == "quit":
            print("Converting to JSON file...")
            break
        print("")
        bounds = input("Enter comma separated bounds, no parentheses, from lowest to highest: ")
        print("")
        param_type = input("Enter type for parameter: int, float, or str.")
        print("")
        if param_type == "int":
            param_type = int
        elif param_type == "float":
            param_type = float
        else:
            print("NOTE: using categorical hyperparameter space.")
            param_type = str
        bounds = bounds.replace(" ", "")
        bounds = bounds.split(",")
        bound_list = []
        for b in bounds:
            bound_list.append(param_type(b))
        bound_list = tuple(bound_list)
        param_dict[param] = bound_list

    if not bool(param_dict):
        print("No parameters provided. Exiting without writing JSON file.")
        sys.exit()
    else:
        converted = json.dumps(param_dict, indent=4)
        if output[-5:] != ".json":
            output += ".json"
        f = open(output, "w")
        f.write(converted)
        f.close()
        print("Wrote json bounds file to " + output + ". Exiting.")
        return


if __name__ == "__main__":
    create_json()
