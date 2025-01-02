from FireBots import FireBots
from MiceBots import MiceBots
from NeuralNetBot import NeuralNetBot
import tkinter as tk
import numpy as np
import random

def main():
    while True:
        menunum = int(input( """
Select simulation:
1) Bots avoid fire and reach fire extinguisher
2) Bots catch mice
3) Mice simulation but bot is a neural net
0) Exit\n"""))
        if menunum == 1:
            run = FireBots()
            run.main()
        if menunum == 2:
            run = MiceBots()
            run.main()
        if menunum == 3:
            run = NeuralNetBot()
            run.main()
        if menunum == 0:
            exit()


if __name__ == '__main__':
    main()