"""
@file main.py
@brief "Main File"
@author Alfred Chavez
"""

from ProductPlanner import ProductPlanner
import json
import pandas as pd
import csv


if __name__ == "__main__":
    print '[+]Loading configuration file'
    with open("config.json", "r") as cfile:
        config = json.load(cfile)
    planner = ProductPlanner(config)
    print '[+]Loading data from csv file'
    planner.add_product('noname','notype',0.0)
    with open(config["data"]) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            planner.add_product(str(row[0]), str(row[1]), float(row[2]))
    print '[+]Start Learning'
    print '[+] * Initialize R-Matrix'
    planner.initialize_matrix()
    print '[+] * Start Monte-Carlo Learning ...'
    planner.monte_carlo()
    print '[+] * Start Q-Learning ...'
    qmatrix = planner.learn()
    print 'Execution Finished'
    cost, data = planner.get_answer()
    print 'Minimum Cost Obtained:', cost
    print 'Actual Budget:', planner.budget
    print 'Data file:', config["data"]
    print 'Data:'
    print 'Products:'
    for i in data:
        print "\tProduct:", i[0], "Product Provider:", i[1][0], "Product Cost:", i[1][1]
