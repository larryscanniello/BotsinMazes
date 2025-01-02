import numpy as np
import random
import math
import heapq
import pandas as pd
import itertools
from collections import deque
from collections import Counter
from statistics import median


class MiceBots:
    def main(self):
        menunum = -1
        np.set_printoptions(suppress=True,threshold=np.inf)
        while not menunum == 0:
            menunum = int(input("""\nSelect mode:
1) Random layout per simulation, random mouse detector sensitivity per simulation
2) Random layout per simulation, custom mouse detector sensitivity fixed for all simulations
3) Single random layout fixed for all simulations, custom mouse detector sensitivity fixed for all simulations                        
4) Overwrite/make blank old shipresults file                            
0) Exit\n"""))
            if menunum == 1 or menunum == 2 or menunum==3:
                self.sim_interface(menunum)
            if menunum == 4:
                file = open('shipresults.txt','w')
                file.write(' ')
                file.close()

    def sim_interface(self,menunum):
        #This is where all of the iterations of the simulation are completed
        iterations = int(input('\nEnter number of times to run test (warning: running many tests may take a long time)\n'))
        printyn = int(input("""\nResults will be printed to terminal at the end.
Enter 1 to visualize all simulations by writing to shipresults.txt
Enter 0 to write nothing to separate file (recommended if many simulations are run)\n"""))
        stoch = int(input('\nEnter 1 to catch stationary mice, 2 to catch stochastic mice\n'))
        if stoch == 2:
            stoch = True
        else:
            stoch = False
        nummice = int(input('\nEnter the number of mice (1 or 2)\n'))
        D = int(input('\nTo make DxD grid, enter D (between 20 and 40 recommended)\n'))
        if menunum == 1 and nummice==1:
            #random alpha, 1 mouse, random grid per trial
            mainresultsarrarr = []
            mainresultsarr = []
            print("\nSimulations have begun.")
            for i in range(iterations):
                newgrid = self.create_grid(D)
                #k is between 4 and 100, but can be changed
                num4coinflip = math.floor(random.random()*96)+4
                a = -math.log(0.5)/(num4coinflip-1)
                runsimresults = self.run_sim_1mouse(newgrid.copy(),a,nummice,stoch,printyn)
                mainresultsarr.append(runsimresults)
                if (i+1)%10 == 0 and i>0:
                    print(i+1,' simulations done')
            mainresultsarrarr.append(mainresultsarr)
        if menunum == 1 and nummice==2:
            #random alpha, 2 mice, random grid per trial
            mainresultsarrarr = []
            mainresultsarr = []
            print("\nSimulations have begun.")
            for i in range(iterations):
                newgrid = self.create_grid(D)
                num4coinflip = math.floor(random.random()*96)+4
                a = -math.log(0.5)/(num4coinflip-1)
                runsimresults = self.run_sim_2mice(newgrid.copy(),a,nummice,stoch,printyn)
                mainresultsarr.append(runsimresults)
                if (i+1)%2 == 0 and i>0:
                    print(i+1,' simulations done')
            mainresultsarrarr.append(mainresultsarr)
        if menunum == 2 and nummice ==1:
            #fixed alpha, random grid per trial, 1 mouse
            while True:
                a = float(input('\nEnter number >0, lower value = higher sensitivity of the mouse detector, recommended range [0.06,0.23]\n'))
                if a>0:
                    break
                print("Error: Number not greater than 0.")
            mainresultsarrarr = []
            mainresultsarr = []
            print("\nSimulations have begun.")
            for i in range(1,iterations+1):
                newgrid = self.create_grid(D)
                runsimresults = self.run_sim_1mouse(newgrid.copy(),a,nummice,stoch,printyn)
                mainresultsarr.append(runsimresults)
                if (i+1)%10 == 0 and i>0:
                    print(i+1,' simulations done')
            mainresultsarrarr.append(mainresultsarr)
        if menunum == 2 and nummice == 2:
            #fixed alpha, random grid per trial, 2 mice
            while True:
                a = float(input('\nEnter number >0, lower value = higher sensitivity of the mouse detector, recommended range [0.06,0.23]\n'))
                if a>0:
                    break
                print("Error: Number not greater than 0.")
            mainresultsarr = []
            print("\nSimulations have begun.")
            for i in range(1,iterations+1):
                newgrid = self.create_grid(D)
                runsimresults = self.run_sim_2mice(newgrid.copy(),a,nummice,stoch,printyn)
                mainresultsarr.append(runsimresults)
                if i%2==0:
                    print(i,' simulations done')
            mainresultsarrarr = [mainresultsarr]
        if menunum == 3 and nummice == 1:
            #fixed alpha, same random grid used for every trial, 1 mouse
            mainresultsarr = []
            while True:
                a = float(input('\nEnter number >0, lower value = higher sensitivity of the mouse detector, recommended range [0.06,0.23]\n'))
                if a>0:
                    break
                print("Error: Number not greater than 0.")
            newgrid = self.create_grid(D)
            print("\nSimulations have begun.")
            for i in range(iterations):
                runsimresults = self.run_sim_1mouse(newgrid.copy(),a,nummice,stoch,printyn)
                mainresultsarr.append(runsimresults)
                if (i+1)%10==0:
                    print(i+1,' simulations done')
            mainresultsarrarr = [mainresultsarr]
        if menunum == 3 and nummice == 2:
            #fixed alpha, same random grid used for every trial, 2 mice
            mainresultsarr = []
            while True:
                a = float(input('\nEnter number >0, lower value = higher sensitivity of the mouse detector, recommended range [0.06,0.23]\n'))
                if a>0:
                    break
                print("Error: Number not greater than 0.")
            newgrid = self.create_grid(D)
            print("\nSimulations have begun.")
            for i in range(iterations):
                runsimresults = self.run_sim_2mice(newgrid.copy(),a,nummice,stoch,printyn)
                mainresultsarr.append(runsimresults)
                if (i+1)%2==0 and i>0:
                    print(i+1,' simulations done')
            mainresultsarrarr = [mainresultsarr]
        for arr in mainresultsarrarr:
            df = pd.DataFrame(arr,index=range(1,len(arr)+1),columns=['Bot 1 mouse 1','Bot 1 mouse 2','Bot 2 mouse 1','Bot 2 mouse 2','Bot 3 mouse 1','Bot 3 mouse 2','Alpha'])
            df.loc['Mean']=df.mean()
            df.loc['Median']=df.median()
            df.loc['Max']=df.max()
            df.loc['Std']=df.std()
            print(df)
        return

    def run_sim_1mouse(self,grid,a,nummice,stoch,printyn):
        grid,botindex,mouseindex,mouseindex2 = self.place_initial_positions(grid,nummice)
        if printyn:
            file = open('shipresults.txt','a')
            file.write('t= 0 \n')
            file.write(np.array2string(grid,max_line_width=1000))
            file.write('\n')
            file.close()
        #everything gets initialized here
        bot1index,bot2index,bot3index = botindex,botindex,botindex
        done = False
        #print(np.array2string(grid,max_line_width=1000))
        estimatedstatedists = self.get_initial_dist(grid)
        bot1eststates,bot2eststates,bot3eststates = estimatedstatedists,estimatedstatedists,estimatedstatedists
        bot1evidence,bot2evidence,bot3evidence = [None],[None],[None]
        #Evidence format: (t,type,tuple of index on ship) 
        #where type = 0 if negative sense, 1 if positive sense, 2 if walked into square"""
        bot1plan,bot2plan,bot3plan = [],[],[]
        b1,b2,b3 = False,False,False
        #these are false if bot is not done, true if bot is done
        t = 1
        runsimresults = [None,None,None,None,None,None,a]
        while b1 == False or b2 == False or b3 == False:
            #this where the individual simulations are run. bot_i moves bot i one step, move_mouse moves mouse a step.
            if not b1:
                grid,bot1index,bot1evidence,bot1plan,bot1eststates = self.bot_1(grid,bot1index,t,bot1plan,bot1evidence.copy(),mouseindex,a,bot1eststates.copy(),stoch)
            if not b2:
                grid,bot2index,bot2evidence,bot2plan,bot2eststates = self.bot_2(grid,bot2index,t,bot2plan,bot2evidence.copy(),mouseindex,a,bot2eststates.copy(),stoch)
            if not b3:
                grid,bot3index,bot3evidence,bot3plan,bot3eststates = self.bot_3(grid,bot3index,t,bot3plan,bot3evidence.copy(),mouseindex,a,bot3eststates.copy(),stoch)
            if stoch:
                grid,mouseindex = self.move_mouse(grid,mouseindex)
            #if bot/mice on same space, not evaluated until everything has moved
            if bot1index == mouseindex and b1==False:
                b1 = True
                grid[(bot1index)]-= 10
                runsimresults[0] = t
            if bot2index == mouseindex and b2==False:
                b2 = True
                grid[(bot2index)]-=100
                runsimresults[2] = t
            if bot3index == mouseindex and b3==False:
                b3 = True
                grid[(bot3index)]-=1000
                runsimresults[4] = t
            if printyn:
                file = open('shipresults.txt','a')
                file.write('t=')
                file.write(str(t))
                file.write('\n')
                file.write(np.array2string(grid,max_line_width=1000))
                file.write('\n')
                file.close()
            #this is where states/grid are printed if option is selected
            #if t goes higher than 5000, simulation is aborted
            if t>5000:
                file = open('shipresults.txt','a')
                file.write(np.array2string(grid,max_line_width=1000))
                file.write('\n')
                if b1 == False:
                    file.write(np.array2string(bot1eststates[t],max_line_width=1000))
                    file.write('\n')
                    runsimresults[0] = 5000
                if b2 == False:
                    file.write(np.array2string(bot2eststates[t],max_line_width=1000))
                    file.write('\n')
                    runsimresults[1] = 5000
                if b3 == False:
                    file.write(np.array2string(bot3eststates[t],max_line_width=1000))
                    file.write('\n')
                    runsimresults[2] = 5000
                print('t>5000,  bot 1: ',b1,' bot 2: ', b2,' bot 3: ',b3)
                return runsimresults
            t += 1
        return runsimresults

    def run_sim_2mice(self,grid,a,nummice,stoch,printyn):
        grid,botindex,mouseindex1,mouseindex2 = self.place_initial_positions(grid,nummice)
        if printyn:
            file = open('shipresults.txt','a')
            file.write('t= 0 \n')
            file.write(np.array2string(grid,max_line_width=1000))
            file.write('\n')
            file.close()
        bot1index,bot2index,bot3index = botindex,botindex,botindex
        done = False
        currentstate,openpairslist = self.get_initial_dist_2mice(grid)
        bot1state,bot2state,bot3state = currentstate.copy(),currentstate.copy(),currentstate.copy()
        bot1evidence,bot2evidence,bot3evidence = [1],[2],[3]
        #Evidence format: (t,type,tuple of index on ship) 
        #where type = 0 if negative sense, 1 if positive sense, 2 if walked into square"""
        bot1plan,bot2plan,bot3plan = [],[],[None,None,None,None]
        bot1mode,bot2mode,bot3mode = 2,2,2
        #botmode = 2 means hasn't caught a mouse, botmode=1 means has caught 1 mouse, botmode = 0 means has caught both mice
        b1secondmouse,b2secondmouse,b3secondmouse = 0,0,0
        t = 1
        runsimresults = [None,None,None,None,None,None,a]

        try:
            #this is where the main simulation is run. analogous to run_sim_1mouse
            while bot1mode>0 or bot2mode>0 or bot3mode>0:
                if bot1mode==2:
                    grid,bot1index,bot1evidence,bot1plan,bot1state,bot1margstate = self.bot_1_2mice(grid,bot1index,t,bot1plan,bot1evidence.copy(),mouseindex1,mouseindex2,a,bot1state.copy(),stoch,openpairslist)
                if bot2mode==2:
                    grid,bot2index,bot2evidence,bot2plan,bot2state = self.bot_2_2mice(grid,bot2index,t,bot2plan,bot2evidence.copy(),mouseindex1,mouseindex2,a,bot2state.copy(),stoch,openpairslist)
                if bot3mode==2:
                    grid,bot3index,bot3evidence,bot3plan,bot3state = self.bot_3_2mice(grid,bot3index,t,bot3plan,bot3evidence.copy(),mouseindex1,mouseindex2,a,bot3state.copy(),stoch,openpairslist)
                
                if bot1mode==1:
                    if b1secondmouse == 1:
                        grid,bot1index,bot1evidence,bot1plan,bot1state1mouse = self.bot_1(grid,bot1index,t,bot1plan,bot1evidence.copy(),mouseindex1,a,bot1state1mouse.copy(),stoch)
                    if b1secondmouse == 2:
                        grid,bot1index,bot1evidence,bot1plan,bot1state1mouse = self.bot_1(grid,bot1index,t,bot1plan,bot1evidence.copy(),mouseindex2,a,bot1state1mouse.copy(),stoch)
                if bot2mode==1:
                    if b2secondmouse == 1:
                        grid,bot2index,bot2evidence,bot2plan,bot2state1mouse = self.bot_2(grid,bot2index,t,bot2plan,bot2evidence.copy(),mouseindex1,a,bot2state1mouse.copy(),stoch)
                    if b2secondmouse == 2:
                        grid,bot2index,bot2evidence,bot2plan,bot2state1mouse = self.bot_2(grid,bot2index,t,bot2plan,bot2evidence.copy(),mouseindex2,a,bot2state1mouse.copy(),stoch)
                if bot3mode==1:
                    if b3secondmouse == 1:
                        grid,bot3index,bot3evidence,bot3plan,bot3state1mouse = self.bot_3(grid,bot3index,t,bot3plan,bot3evidence.copy(),mouseindex1,a,bot3state1mouse.copy(),stoch)
                    if b3secondmouse == 2:
                        grid,bot3index,bot3evidence,bot3plan,bot3state1mouse = self.bot_3(grid,bot3index,t,bot3plan,bot3evidence.copy(),mouseindex2,a,bot3state1mouse.copy(),stoch)
                if stoch:
                    grid,mouseindex1,mouseindex2 = self.move_mice(grid,mouseindex1,mouseindex2,bot1mode,bot2mode,bot3mode,b1secondmouse,b2secondmouse,b3secondmouse)
                #the above is where the main movement of bots /mice happen
                #below is just catching all possible combinations of bot i catching mouse j, adjusting mode accordingly, keeping track of which mouse is caught
                if bot1mode==2 and (bot1index==mouseindex1 or bot1index==mouseindex2):
                    bot1mode=1
                    if bot1index==mouseindex1:
                        bot1mode=1
                        b1secondmouse=2
                        grid[bot1index]+=10
                        bot1state1mouse = self.get_marginal_dist_for_index(bot1state.copy(),mouseindex1,openpairslist,len(grid),grid.copy(),False)
                        #print(np.array2string(bot1state1mouse,max_line_width=1000))
                    if bot1index==mouseindex2:
                        bot1mode=1
                        b1secondmouse=1
                        grid[bot1index]+=10
                        bot1state1mouse = self.get_marginal_dist_for_index(bot1state.copy(),mouseindex2,openpairslist,len(grid),grid.copy(),False)
                    runsimresults[0]=t
                    bot1plan = []
                if bot2mode==2 and (bot2index==mouseindex1 or bot2index==mouseindex2):
                    bot2mode=1
                    if bot2index==mouseindex1:
                        b2secondmouse=2
                        grid[bot2index]+=100
                        bot2state1mouse = self.get_marginal_dist_for_index(bot2state.copy(),mouseindex1,openpairslist,len(grid),grid.copy(),False)
                        #print(np.array2string(bot1state1mouse,max_line_width=1000))
                    if bot2index==mouseindex2:
                        bot2mode=1
                        b2secondmouse=1
                        grid[bot2index]+=100
                        bot2state1mouse = self.get_marginal_dist_for_index(bot2state.copy(),mouseindex2,openpairslist,len(grid),grid.copy(),False)
                    runsimresults[2]=t
                    bot2plan = []
                if bot3mode==2 and (bot3index==mouseindex1 or bot3index==mouseindex2):    
                    bot3mode=1
                    if bot3index==mouseindex1:
                        b3secondmouse=2
                        grid[bot3index]+=1000
                        bot3state1mouse = self.get_marginal_dist_for_index(bot2state.copy(),mouseindex1,openpairslist,len(grid),grid.copy(),False)
                        #print(np.array2string(bot1state1mouse,max_line_width=1000))
                    if bot3index==mouseindex2:
                        bot3mode=1
                        b3secondmouse=1
                        grid[bot3index]+=1000
                        bot3state1mouse = self.get_marginal_dist_for_index(bot2state.copy(),mouseindex2,openpairslist,len(grid),grid.copy(),False)
                    runsimresults[4]=t
                    bot3plan = []
                    
                if bot1mode==1 and ((b1secondmouse==1 and bot1index==mouseindex1) or (b1secondmouse==2 and bot1index==mouseindex2)):
                    grid[bot1index]-=10
                    if b1secondmouse==1:
                        grid[mouseindex2]-=10
                    else:
                        grid[mouseindex1]-=10
                    runsimresults[1] = t
                    bot1mode = 0
                if bot2mode==1 and ((b2secondmouse==1 and bot2index==mouseindex1) or (b2secondmouse==2 and bot2index==mouseindex2)):
                    grid[bot2index]-=100
                    if b2secondmouse==1:
                        grid[mouseindex2]-=100
                    else:
                        grid[mouseindex1]-=100
                    runsimresults[3] = t
                    bot2mode = 0
                if bot3mode==1 and ((b3secondmouse==1 and bot3index==mouseindex1) or (b3secondmouse==2 and bot3index==mouseindex2)):
                    grid[bot3index]-=1000
                    if b3secondmouse==1:
                        grid[mouseindex2]-=1000
                    else:
                        grid[mouseindex1]-=1000
                    runsimresults[5] = t
                    bot3mode = 0
                if printyn:
                    file = open('shipresults.txt','a')
                    file.write('t= ')
                    file.write(str(t))
                    file.write('\n')
                    file.write(np.array2string(grid, max_line_width=1000))
                    file.write('\n')
                    file.close()
                if t>2500:
                    #aborts after 2500 time frames
                    if bot1mode>0:
                        runsimresults[1] = 2500
                        bot1mode=0
                    if bot2mode>0:
                        runsimresults[3] = 2500
                        bot2mode=0
                    if bot3mode>0:
                        runsimresults[5] = 2500
                        bot3mode=0
                t += 1
        except:
            print('Iteration skipped.')
            return [None,None,None,None,None,None,a]
        return runsimresults

    def bot_1_2mice(self,grid,bot1index,t,plan,bot1evidence,mouseindex1,mouseindex2,a,bot1state,stoch,openpairslist):
        #plan is a list of tuples. if empty, then bot will sense and get new plan
        if plan == []:
            d1 = self.calc_manhattan_dist(bot1index,mouseindex1)
            d2 = self.calc_manhattan_dist(bot1index,mouseindex2)
            if random.random()<1-((1-math.exp(-a*(d1-1)))*(1-math.exp(-a*(d2-1)))):
                newevidence = (t,1,bot1index)
                bot1evidence.append(newevidence)
            else:
                newevidence = (t,0,bot1index)
                bot1evidence.append(newevidence)
            bot1state = self.filtering_2mice(bot1state.copy(),stoch,bot1evidence.copy(),t,grid.copy(),a,openpairslist)
            probmouseat = self.calc_marginal_dists(bot1state.copy(),openpairslist,len(grid))
            destinationindex = np.unravel_index(probmouseat.argmax(), probmouseat.shape)
            plan = self.bfs(grid,bot1index,destinationindex,1)
            if plan == []:
                print('Bot 1 empty plan')
            return grid,bot1index,bot1evidence,plan,bot1state,probmouseat
        
        else:
            #if plan still has tuple in it, then bot will pop first one and go to that index
            grid[bot1index]-=10
            grid[plan[0]]+=10
            bot1index = plan[0]
            plan.pop(0)
            newevidence = (t,2,bot1index)
            bot1evidence.append(newevidence)
            bot1state = self.filtering_2mice(bot1state.copy(),stoch,bot1evidence.copy(),t,grid.copy(),a,openpairslist)
            probmouseat = self.calc_marginal_dists(bot1state.copy(),openpairslist,len(grid))
            return grid,bot1index,bot1evidence,plan,bot1state,probmouseat
    def bot_2_2mice(self,grid,bot2index,t,plan,bot2evidence,mouseindex1,mouseindex2,a,bot2state,stoch,openpairslist):
        #like bot 1, if plan is empty then time to sense and get new plan
        if plan == []:
            d1 = self.calc_manhattan_dist(bot2index,mouseindex1)
            d2 = self.calc_manhattan_dist(bot2index,mouseindex2)
            if random.random()<1-((1-math.exp(-a*(d1-1)))*(1-math.exp(-a*(d2-1)))):
                newevidence = (t,1,bot2index)
                bot2evidence.append(newevidence)
            else:
                newevidence = (t,0,bot2index)
                bot2evidence.append(newevidence)
            bot2state = self.filtering_2mice(bot2state.copy(),stoch,bot2evidence.copy(),t,grid,a,openpairslist)
            probmouseat = self.calc_marginal_dists(bot2state,openpairslist,len(grid))
            destinationindex = np.unravel_index(probmouseat.argmax(), probmouseat.shape)
            #if destination is the same as current bot index, then find 2nd highest probability
            if bot2index == destinationindex:
                statecopy = probmouseat.copy()
                statecopy[bot2index]-=1
                destinationindex = np.unravel_index(statecopy.argmax(), statecopy.shape)
            plan = self.bfs(grid,bot2index,destinationindex,2)
            return grid,bot2index,bot2evidence,plan,bot2state
        if plan[0]==None:
            #the plan for bot 2 is of the form [tup, None, tup, None, tup,...] and if the current plan is None, that means to sense
            d1 = self.calc_manhattan_dist(bot2index,mouseindex1)
            d2 = self.calc_manhattan_dist(bot2index,mouseindex2)
            if random.random()<1-((1-math.exp(-a*(d1-1)))*(1-math.exp(-a*(d2-1)))):
                newevidence = (t,1,bot2index)
                bot2evidence.append(newevidence)
            else:
                newevidence = (t,0,bot2index)
                bot2evidence.append(newevidence)
            bot2state = self.filtering_2mice(bot2state.copy(),stoch,bot2evidence.copy(),t,grid,a,openpairslist)
            probmouseat = self.calc_marginal_dists(bot2state,openpairslist,len(grid))
            plan.pop(0)
            destinationindex = np.unravel_index(probmouseat.argmax(), probmouseat.shape)
            if bot2index == destinationindex:
                statecopy = probmouseat.copy()
                statecopy[bot2index]-=1
                destinationindex = np.unravel_index(statecopy.argmax(), statecopy.shape)
            #replans at every step
            if not plan[len(plan)-1] == destinationindex:
                plan = self.bfs(grid,bot2index,destinationindex,2)
            return grid,bot2index,bot2evidence,plan,bot2state
        else:
            #if the current plan is not to sense, then moves to planned index
            grid[bot2index]-=100
            grid[plan[0]]+=100
            bot2index = plan[0]
            plan.pop(0)
            newevidence = (t,2,bot2index)
            bot2evidence.append(newevidence)
            bot2state = self.filtering_2mice(bot2state.copy(),stoch,bot2evidence.copy(),t,grid,a,openpairslist)
            return grid,bot2index,bot2evidence,plan,bot2state
        return
    def bot_3_2mice(self,grid,bot3index,t,plan,bot3evidence,mouseindex1,mouseindex2,a,bot3state,stoch,openpairslist):
        #if plan is empty, sense and get new plan
        if plan == []:
            d1 = self.calc_manhattan_dist(bot3index,mouseindex1)
            d2 = self.calc_manhattan_dist(bot3index,mouseindex2)
            if random.random()<1-((1-math.exp(-a*(d1-1)))*(1-math.exp(-a*(d2-1)))):
                newevidence = (t,1,bot3index)
                bot3evidence.append(newevidence)
            else:
                newevidence = (t,0,bot3index)
                bot3evidence.append(newevidence)
            bot3state = self.filtering_2mice(bot3state.copy(),stoch,bot3evidence.copy(),t,grid.copy(),a,openpairslist)
            probmouseat = self.calc_marginal_dists(bot3state.copy(),openpairslist,len(grid))
            destinationindex = np.unravel_index(probmouseat.argmax(), probmouseat.shape)
            if bot3index == destinationindex:
                statecopy = probmouseat.copy()
                statecopy[bot3index]-=1
                destinationindex = np.unravel_index(statecopy.argmax(),statecopy.shape)
            if stoch:
                #explained in write-up; if close to target then doesn't alternate sense and move. if stationary, always alternate
                estd = self.calc_manhattan_dist(bot3index,destinationindex)
                if estd>3:
                    plan = self.bot_3_dynamic_UFCS_2mice(grid.copy(),bot3index,destinationindex,bot3state.copy(),3,False,bot3evidence.copy(),t)
                    return grid,bot3index,bot3evidence,plan,bot3state
                else:
                    plan = self.bot_3_dynamic_UFCS_2mice(grid.copy(),bot3index,destinationindex,bot3state.copy(),1,True,bot3evidence.copy(),t)
                    return grid,bot3index,bot3evidence,plan,bot3state
            else:
                plan = self.bot_3_dynamic_UFCS_2mice(grid.copy(),bot3index,destinationindex,bot3state.copy(),3,False,bot3evidence.copy(),t)
                return grid,bot3index,bot3evidence,plan,bot3state
        #plan is of the form [tup,None,tup,None,tup,...] if current plan is None, that means sense
        if plan[0]==None:
            d1 = self.calc_manhattan_dist(bot3index,mouseindex1)
            d2 = self.calc_manhattan_dist(bot3index,mouseindex2)
            if random.random()<1-((1-math.exp(-a*(d1-1)))*(1-math.exp(-a*(d2-1)))):
                newevidence = (t,1,bot3index)
                bot3evidence.append(newevidence)
            else:
                newevidence = (t,0,bot3index)
                bot3evidence.append(newevidence)
            bot3state= self.filtering_2mice(bot3state.copy(),stoch,bot3evidence.copy(),t,grid,a,openpairslist)
            plan.pop(0)
            return grid,bot3index,bot3evidence,plan,bot3state
        else:
            #if plan is a tup, then move there
            grid[bot3index]-=1000
            grid[plan[0]]+=1000
            bot3index = plan[0]
            plan.pop(0)
            newevidence = (t,2,bot3index)
            bot3evidence.append(newevidence)
            bot3state = self.filtering_2mice(bot3state.copy(),stoch,bot3evidence.copy(),t,grid,a,openpairslist)
            return grid,bot3index,bot3evidence,plan,bot3state

    def move_mouse(self,grid,mouseindex):
        #choose a place to move to or stay in place
        adjlist = self.get_adj_indices(mouseindex[0],mouseindex[1],len(grid))
        openadjlist = []
        for index in adjlist:
            if grid[index]%10==0:
                openadjlist.append(index)
        randomint = math.floor(random.random()*(len(openadjlist)+1))
        if randomint == len(openadjlist):
            return grid,mouseindex
        else:
            grid[mouseindex]-=2
            grid[openadjlist[randomint]]+=2
            mouseindex = openadjlist[randomint]
            return grid,mouseindex
    def move_mice(self,grid,mouseindex1,mouseindex2,bot1mode,bot2mode,bot3mode,b1secondmouse,b2secondmouse,b3secondmouse):
        #finds all possible index pairs to be in, chooses one randomly. Impossible for mice to be in same spot
        adjlist1 = self.get_adj_indices(mouseindex1[0],mouseindex1[1],len(grid))
        adjlist2 = self.get_adj_indices(mouseindex2[0],mouseindex2[1],len(grid))
        openadjlist1,openadjlist2 = [],[]
        for index1 in adjlist1:
            if grid[index1]%10==0 or grid[index1]%10==2:
                openadjlist1.append(index1)
        for index2 in adjlist2:
            if grid[index2]%10==0 or grid[index2]%10==2:
                openadjlist2.append(index2)
        openadjlist1.append(mouseindex1)
        openadjlist2.append(mouseindex2)
        possiblemoves = [[i,j] for i,j in itertools.product(openadjlist1,openadjlist2) if not i==j]
        movetopick = math.floor(random.random()*len(possiblemoves))
        mouse1 = grid[mouseindex1]
        mouse2 = grid[mouseindex2]
        grid[mouseindex1]-=2
        grid[mouseindex2]-=2
        grid[possiblemoves[movetopick][0]]+=2
        grid[possiblemoves[movetopick][1]]+=2
        #this next code is in case a mouse has been caught by a bot, to keep track of that bot
        if bot1mode==1 and b1secondmouse==2:
            grid[mouseindex1]-=10
            grid[possiblemoves[movetopick][0]]+=10
        if bot1mode==1 and b1secondmouse==1:
            grid[mouseindex2]-=10
            grid[possiblemoves[movetopick][1]]+=10
        if bot2mode==1 and b2secondmouse==2:
            grid[mouseindex1]-=100
            grid[possiblemoves[movetopick][0]]+=100
        if bot2mode==1 and b2secondmouse==1:
            grid[mouseindex2]-=100
            grid[possiblemoves[movetopick][1]]+=100
        if bot3mode==1 and b3secondmouse==2:
            grid[mouseindex1]-=1000
            grid[possiblemoves[movetopick][0]]+=1000
        if bot3mode==1 and b3secondmouse==1:
            grid[mouseindex2]-=1000
            grid[possiblemoves[movetopick][1]]+=1000
        mouseindex1 = possiblemoves[movetopick][0]
        mouseindex2 = possiblemoves[movetopick][1]
        return grid,mouseindex1,mouseindex2

    #bot_i works the same as bot_i_2mice
    def bot_1(self,grid,bot1index,t,plan,bot1evidence,mouseindex,a,bot1state,stoch):
        if plan == []:
            d = self.calc_manhattan_dist(bot1index,mouseindex)
            if random.random()<math.exp(-a*(d-1)):
                newevidence = (t,1,bot1index)
                bot1evidence.append(newevidence)
            else:
                newevidence = (t,0,bot1index)
                bot1evidence.append(newevidence)
            bot1state = self.filtering(bot1state.copy(),stoch,bot1evidence.copy(),t,grid.copy(),a)
            destinationindex = np.unravel_index(bot1state.argmax(), bot1state.shape)
            plan = self.bfs(grid,bot1index,destinationindex,1)
            return grid,bot1index,bot1evidence,plan,bot1state
        else:
            grid[bot1index]-=10
            grid[plan[0]]+=10
            bot1index = plan[0]
            plan.pop(0)
            newevidence = (t,2,bot1index)
            bot1evidence.append(newevidence)
            bot1state = self.filtering(bot1state.copy(),stoch,bot1evidence.copy(),t,grid.copy(),a)
            return grid,bot1index,bot1evidence,plan,bot1state 
    def bot_2(self,grid,bot2index,t,plan,bot2evidence,mouseindex,a,bot2state,stoch):
        if plan == []:
            d = self.calc_manhattan_dist(bot2index,mouseindex)
            if random.random()<math.exp(-a*(d-1)):
                newevidence = (t,1,bot2index)
                bot2evidence.append(newevidence)
            else:
                newevidence = (t,0,bot2index)
                bot2evidence.append(newevidence)
            bot2state = self.filtering(bot2state.copy(),stoch,bot2evidence.copy(),t,grid,a)
            destinationindex = np.unravel_index(bot2state.argmax(), bot2state.shape)
            if bot2index == destinationindex:
                statecopy = bot2state.copy()
                statecopy[bot2index]-=1
                destinationindex = np.unravel_index(statecopy.argmax(), bot2state.shape)
            plan = self.bfs(grid,bot2index,destinationindex,2)
            return grid,bot2index,bot2evidence,plan,bot2state
        if plan[0]==None:
            d = self.calc_manhattan_dist(bot2index,mouseindex)
            if random.random()<math.exp(-a*(d-1)):
                newevidence = (t,1,bot2index)
                bot2evidence.append(newevidence)
            else:
                newevidence = (t,0,bot2index)
                bot2evidence.append(newevidence)
            bot2state = self.filtering(bot2state.copy(),stoch,bot2evidence.copy(),t,grid,a)
            plan.pop(0)
            destinationindex = np.unravel_index(bot2state.argmax(), bot2state.shape)
            if not plan[len(plan)-1] == destinationindex:
                plan = self.bfs(grid,bot2index,destinationindex,2)
            return grid,bot2index,bot2evidence,plan,bot2state
        else:
            grid[bot2index]-=100
            grid[plan[0]]+=100
            bot2index = plan[0]
            plan.pop(0)
            newevidence = (t,2,bot2index)
            bot2evidence.append(newevidence)
            bot2state= self.filtering(bot2state.copy(),stoch,bot2evidence.copy(),t,grid,a)
            return grid,bot2index,bot2evidence,plan,bot2state
    def bot_3(self,grid,bot3index,t,plan,bot3evidence,mouseindex,a,bot3state,stoch):
        if plan == []:
            d = self.calc_manhattan_dist(bot3index,mouseindex)
            if random.random()<math.exp(-a*(d-1)):
                newevidence = (t,1,bot3index)
                bot3evidence.append(newevidence)
            else:
                newevidence = (t,0,bot3index)
                bot3evidence.append(newevidence)
            bot3state = self.filtering(bot3state.copy(),stoch,bot3evidence.copy(),t,grid.copy(),a)
            destinationindex = np.unravel_index(bot3state.argmax(), bot3state.shape)
            if bot3index == destinationindex:
                statecopy = bot3state.copy()
                statecopy[bot3index]-=1
                destinationindex = np.unravel_index(statecopy.argmax(), bot3state.shape)
            if stoch:
                estd = self.calc_manhattan_dist(bot3index,destinationindex)
                if estd>3:
                    plan = self.bot_3_dynamic_UFCS(grid,bot3index,destinationindex,bot3state.copy(),3,stoch,bot3evidence.copy(),t)
                    return grid,bot3index,bot3evidence,plan,bot3state
                else:
                    plan = self.bot_3_dynamic_UFCS(grid,bot3index,destinationindex,bot3state.copy(),1,stoch,bot3evidence.copy(),t)
                    return grid,bot3index,bot3evidence,plan,bot3state
            else:
                plan = self.bot_3_dynamic_UFCS(grid,bot3index,destinationindex,bot3state.copy(),3,stoch,bot3evidence.copy(),t)
                return grid,bot3index,bot3evidence,plan,bot3state
        if plan[0]==None:
            d = self.calc_manhattan_dist(bot3index,mouseindex)
            if random.random()<math.exp(-a*(d-1)):
                newevidence = (t,1,bot3index)
                bot3evidence.append(newevidence)
            else:
                newevidence = (t,0,bot3index)
                bot3evidence.append(newevidence)
            bot3state= self.filtering(bot3state.copy(),stoch,bot3evidence.copy(),t,grid,a)
            plan.pop(0)
            return grid,bot3index,bot3evidence,plan,bot3state
        else:
            grid[bot3index]-=1000
            grid[plan[0]]+=1000
            bot3index = plan[0]
            plan.pop(0)
            newevidence = (t,2,bot3index)
            bot3evidence.append(newevidence)
            bot3state = self.filtering(bot3state.copy(),stoch,bot3evidence.copy(),t,grid,a)
            return grid,bot3index,bot3evidence,plan,bot3state

    def get_initial_dist(self,grid):
        #returns intial distribution where the mouse is equally likely to be in any open space
        openlist = []
        for i in range(len(grid)):
            for j in range(len(grid)):
                if grid[(i,j)]==0 or grid[(i,j)]==2:
                    openlist.append((i,j))
        initial_dist = np.zeros((len(grid),len(grid)))
        for item in openlist:
            initial_dist[item] = 1/len(openlist)
        #print(np.array2string(initial_dist,max_line_width=1000))
        return initial_dist
    def get_initial_dist_2mice(self,grid):
        #returns a list of samples, where every open pair has the same chance of being sampled
        openlist = []
        l = len(grid)
        for i in range(l):
            for j in range(l):
                if grid[(i,j)]==0 or grid[(i,j)]==2:
                    openlist.append((i,j))
        numopenpairs = len(openlist)*(len(openlist)+1)/2
        twomicearr = np.zeros((l,l,l,l))
        #the pair (i,j), (a,b) is stored as (i,j,a,b) if (i,j)<(a,b), or (a,b,i,j) if (a,b)<(i,j)
        #i believe twomicearr is dead code
        openpairslist = []
        for i in range(l**2):
            for j in range(i,l**2):
                if (grid[(i//l,i%l)]==0 or grid[(i//l,i%l)]==2) and (grid[(j//l,j%l)]==0 or grid[(j//l,j%l)]==2):
                    twomicearr[(i//l,i%l,j//l,j%l)]+=1/numopenpairs
                    openpairslist.append((i//l,i%l,j//l,j%l))
        samplelist = []
        if len(grid)<=20:
            samples = 576*len(grid)-576*20+10000
        else:
            samples = 4500*len(grid)-4500*20+10000
        for i in range(samples):
            indextopick = math.floor(random.random()*len(openpairslist))
            samplelist.append(openpairslist[indextopick])
        return samplelist,openpairslist

    #filtering is explained in-depth in my write-up
    def filtering(self,filterstate,stoch,filterevidence,t,grid,a):
        #First we find P(X_t|e_{t-1},...,e_1), the prediction of the new state from old evidence
        if not stoch:
            pred = filterstate
        else:
            pred = np.zeros((len(grid),len(grid)))
            for i in range(len(grid)):
                for j in range(len(grid)):
                    if grid[(i,j)]%10==0 or grid[(i,j)]%10==2 and not (i,j)==filterevidence[t][2]:
                        adjlist = self.get_adj_indices(i,j,len(grid))
                        openadjlist = []
                        for index in adjlist:
                            if grid[index]%10==0 or grid[index]%10==2 and not (i,j)==filterevidence[t][2]:
                                openadjlist.append(index)
                        temppred = np.zeros((len(grid),len(grid)))
                        temppred[(i,j)]+=1/(len(openadjlist)+1)
                        if len(openadjlist)>0:
                            for index in openadjlist:
                                temppred[index]+=1/(len(openadjlist)+1)
                        pred += temppred*filterstate[(i,j)]
        #Now we need a second array: P(e_t|X_t), the probability of getting our evidence conditioned on the state
        #which will be different for all 3 evidence types
        probevidence = np.zeros((len(grid),len(grid)))
        #Next line is if new evidence is stepping into an empty square, thereby that empty square does not contain the mouse:
        if filterevidence[t][1]==2:
            for i in range(len(grid)):
                for j in range(len(grid)):
                    if (grid[(i,j)]%10==0 or grid[i,j]%10==2) and not (i,j)==filterevidence[t][2]:
                        probevidence[(i,j)] += 1
        #If new evidence is a positive sense:
        if filterevidence[t][1]==1:
            for i in range(len(grid)):
                for j in range(len(grid)):
                    if (grid[(i,j)]%10==0 or grid[i,j]%10==2) and not (i,j)==filterevidence[t][2]:
                        d = self.calc_manhattan_dist((i,j),filterevidence[t][2])
                        probevidence[(i,j)]=math.exp(-a*(d-1))
        #If new evidence is a negative sense:
        if filterevidence[t][1]==0:
            for i in range(len(grid)):
                for j in range(len(grid)):
                    if (grid[(i,j)]%10==0 or grid[i,j]%10==2) and not (i,j)==filterevidence[t][2]:
                        d = self.calc_manhattan_dist((i,j),filterevidence[t][2])
                        probevidence[(i,j)]=1-math.exp(-a*(d-1))
        stateunnormalized = pred * probevidence
        #Now we normalize probabilities
        newstate = stateunnormalized*(1/np.sum(stateunnormalized))
        return newstate
    def filtering_2mice(self,filterstate,stoch,filterevidence,t,grid,a,openpairslist):
        #Sample from P(X_{t+1}|X_t=S[i])
        weights = np.zeros(len(filterstate))
        if stoch:
            for i in range(len(filterstate)):
                adjlist1 = self.get_adj_indices(filterstate[i][0],filterstate[i][1],len(grid))
                adjlist2 = self.get_adj_indices(filterstate[i][2],filterstate[i][3],len(grid))
                openadjlist1,openadjlist2 = [],[]
                for adjindex1 in adjlist1:
                    if grid[adjindex1]%10==0 or grid[adjindex1]%10==2:
                        openadjlist1.append(adjindex1)
                for adjindex2 in adjlist2:
                    if grid[adjindex2]%10==0 or grid[adjindex2]%10==2:
                        openadjlist2.append(adjindex2)
                listofadjpairs = [(i[0],i[1],j[0],j[1]) for i,j in itertools.product(openadjlist1,openadjlist2) if i <= j]
                listofadjpairs += [(j[0],j[1],i[0],i[1]) for i,j in itertools.product(openadjlist1,openadjlist2) if i>j]
                indextopick = math.floor(len(listofadjpairs)*random.random())
                filterstate[i]=listofadjpairs[indextopick]
                if filterevidence[t][1]==2:
                    if not (filterstate[i][0],filterstate[i][1])==filterevidence[t][2] and not (filterstate[i][2],filterstate[i][3])==filterevidence[t][2]:
                        weights[i] = 1
                if filterevidence[t][1]==1:
                    if not (filterstate[i][0],filterstate[i][1])==filterevidence[t][2] and not (filterstate[i][2],filterstate[i][3])==filterevidence[t][2]:
                        d1 = self.calc_manhattan_dist((filterstate[i][0],filterstate[i][1]),filterevidence[t][2])
                        d2 = self.calc_manhattan_dist((filterstate[i][2],filterstate[i][3]),filterevidence[t][2])
                        weights[i]= 1-(1-math.exp(-a*(d1-1)))*(1-math.exp(-a*(d2-1)))
                if filterevidence[t][1]==0:
                    if not (filterstate[i][0],filterstate[i][1])==filterevidence[t][2] and not (filterstate[i][2],filterstate[i][3])==filterevidence[t][2]:
                        d1 = self.calc_manhattan_dist((filterstate[i][0],filterstate[i][1]),filterevidence[t][2])
                        d2 = self.calc_manhattan_dist((filterstate[i][2],filterstate[i][3]),filterevidence[t][2])
                        weights[i]= (1-math.exp(-a*(d1-1)))*(1-math.exp(-a*(d2-1)))
        else:
            for i in range(len(filterstate)):
                if filterevidence[t][1]==2:
                    if not (filterstate[i][0],filterstate[i][1])==filterevidence[t][2] and not (filterstate[i][2],filterstate[i][3])==filterevidence[t][2]:
                        weights[i] = 1
                if filterevidence[t][1]==1:
                    if not (filterstate[i][0],filterstate[i][1])==filterevidence[t][2] and not (filterstate[i][2],filterstate[i][3])==filterevidence[t][2]:
                        d1 = self.calc_manhattan_dist((filterstate[i][0],filterstate[i][1]),filterevidence[t][2])
                        d2 = self.calc_manhattan_dist((filterstate[i][2],filterstate[i][3]),filterevidence[t][2])
                        weights[i]= 1-(1-math.exp(-a*(d1-1)))*(1-math.exp(-a*(d2-1)))
                if filterevidence[t][1]==0:
                    if not (filterstate[i][0],filterstate[i][1])==filterevidence[t][2] and not (filterstate[i][2],filterstate[i][3])==filterevidence[t][2]:
                        d1 = self.calc_manhattan_dist((filterstate[i][0],filterstate[i][1]),filterevidence[t][2])
                        d2 = self.calc_manhattan_dist((filterstate[i][2],filterstate[i][3]),filterevidence[t][2])
                        weights[i]= 1-math.exp(-a*(d1-1))*(1-math.exp(-a*(d2-1)))
        filterstate = self.direct_sampling(len(filterstate),filterstate.copy(),weights.copy())
        if not stoch:
            margdist = self.calc_marginal_dists(filterstate,None,len(grid))
            filterevidence.pop(0)
            beento = [tup for (a,b,tup) in filterevidence]
            for i in range(len(grid)):
                for j in range(len(grid)):
                    if margdist[(i,j)]==0 and (i,j) not in beento and (grid[(i,j)]%10==0 or grid[(i,j)]%10==2):
                        filterstate.append((i,j,i,j))
        return filterstate

    #sample using np.choices
    def direct_sampling(self,N,samples,weights):
        weights*=1/np.sum(weights)
        return random.choices(population=samples,weights=weights,k=N)

    #given distribution of pairs of spaces, returns distribution of probability mouse is at individual spaces via marginalization
    def calc_marginal_dists(self,state,openpairlist,D):
        probmouseat = np.zeros((D,D))
        for i in range(len(state)):
            probmouseat[(state[i][0],state[i][1])]+=1/len(state)
            if not (state[i][0],state[i][1])==(state[i][2],state[i][3]):
                probmouseat[(state[i][2],state[i][3])]+=1/len(state)
        probmouseat*=1/np.sum(probmouseat)
        return probmouseat

    #predicting explained in detail in write-up
    def predicting(self,initialstate,stoch,evidence,grid):
        if not stoch:
            return initialstate
        predictedstate = np.zeros((len(grid),len(grid)))
        for i in range(len(grid)):
            for j in range(len(grid)):
                if (grid[(i,j)]%10==0 or grid[(i,j)]%10==2) and not (i,j)==evidence[len(evidence)-1][2]:
                    adjlist = self.get_adj_indices(i,j,len(grid))
                    openadjlist = []
                    for index in adjlist:
                        if grid[index]%10==0 or grid[index]%10==2 and not index==evidence[len(evidence)-1][2]:
                            openadjlist.append(index)
                    temppred = np.zeros((len(grid),len(grid)))
                    temppred[(i,j)]+=1/(len(openadjlist)+1)
                    if len(openadjlist)>0:
                        for index in openadjlist:
                            temppred[index]+=1/(len(openadjlist)+1)
                    predictedstate += temppred*initialstate[(i,j)]
        return predictedstate
    def predicting_2mice(self,initialstate,shortdist,evidence,grid):
        newlist = []
        if not shortdist:
            return self.calc_marginal_dists(initialstate,None,len(grid)),initialstate
        for i in range(len(initialstate)):
            adjlist1 = self.get_adj_indices(initialstate[i][0],initialstate[i][1],len(grid))
            adjlist2 = self.get_adj_indices(initialstate[i][2],initialstate[i][3],len(grid))
            openadjlist1,openadjlist2 = [],[]
            for adjindex1 in adjlist1:
                if grid[adjindex1]%10==0 or grid[adjindex1]%10==2:
                    openadjlist1.append(adjindex1)
            for adjindex2 in adjlist2:
                if grid[adjindex2]%10==0 or grid[adjindex2]%10==2:
                    openadjlist2.append(adjindex2)
            listofadjpairs = [(i[0],i[1],j[0],j[1]) for i,j in itertools.product(openadjlist1,openadjlist2) if i <= j]
            listofadjpairs += [(j[0],j[1],i[0],i[1]) for i,j in itertools.product(openadjlist1,openadjlist2) if i>j]
            indextopick = math.floor(len(listofadjpairs)*random.random())
            newlist.append(listofadjpairs[indextopick])
        return self.calc_marginal_dists(newlist,None,len(grid)),newlist

    def bot_3_dynamic_UFCS(self,grid,start,destinationindex,state,bot,stoch,evidence,t):
        #turn 3d UFCS into 2d by making weights list of arrays into a hashmap/graph that UFCS algorithm can traverse
        newweights = {(0,start,0):None}
        seen = []
        currlevel = [(0,start,0)]
        j=0
        rowsign = destinationindex[0]-start[0]
        colsign = destinationindex[1]-start[1]
        check = False
        predictedstate = state
        while not currlevel==[]:
            predictedstate = self.predicting(predictedstate.copy(),stoch,evidence.copy(),grid)
            nextlevel = []
            for item in currlevel:
                newweights[item] = None
                if item[1] == destinationindex:
                    check = True
                    break
                adjsqrs = self.get_adj_indices(item[1][0],item[1][1],len(grid))
                for sq in adjsqrs:
                    if (grid[sq]%10==0 or grid[sq]%10==2) and not sq in seen:
                        #These next four lines are made so that if the destination is up and to the right of the start for instance,
                        #the UFCS graph can only point up or right
                        if (sq[0]-item[1][0]==0 or (sq[0]-item[1][0]>0 and rowsign>0) or (sq[0]-item[1][0]<0 and rowsign <0)):
                            if (sq[1]-item[1][1]==0 or (sq[1]-item[1][1]>0 and colsign>0) or (sq[1]-item[1][1]<0 and colsign<0)):
                                if min(start[0],destinationindex[0])<=sq[0]<=max(start[0],destinationindex[0]):
                                    if min(start[1],destinationindex[1])<=sq[1]<=max(start[1],destinationindex[1]):
                                        new = (1+math.log(1-predictedstate[sq]),sq,item[2]+1)
                                        if not new in nextlevel:
                                            nextlevel.append(new)
                                        if newweights[item]==None:
                                            newweights[item]=[new]
                                        else:
                                            newweights[item].append(new)
                seen.append(item[1])
            if check == True:
                break
            currlevel = nextlevel
            j+=1
        check = False
        for item in list(newweights.values()):
            if not item == None:
                for item2 in item:
                    if item2[1]==destinationindex:
                        check = True
        # if no path of just moving in two directions is available, then this happens
        if not check:
            newweights = {(0,start,0):None}
            seen = []
            currlevel = [(0,start,0)]
            j=0
            check = False
            predictedstate = state
            while not currlevel==[]:
                predictedstate = self.predicting(predictedstate.copy(),stoch,evidence.copy(),grid)
                nextlevel = []
                for item in currlevel:
                    newweights[item] = None
                    if item[1] == destinationindex:
                        check = True
                        break
                    adjsqrs = self.get_adj_indices(item[1][0],item[1][1],len(grid))
                    for sq in adjsqrs:
                        if (grid[sq]%10==0 or grid[sq]%10==2) and not sq in seen:
                            new = (1+math.log(1-predictedstate[sq]),sq,item[2]+1)
                            if not new in nextlevel:
                                nextlevel.append(new)
                            if newweights[item]==None:
                                newweights[item]=[new]
                            else:
                                newweights[item].append(new)
                    seen.append(item[1])
                if check == True:
                    break
                currlevel = nextlevel
                j+=1
        #print(newweights)
        distances = {}
        prev = {}
        fringe = []
        distances[(0,start,0)]=0
        prev[start]= None
        #UFCS on graph
        heapq.heappush(fringe,(0,start,0))
        while not fringe == []:
            curr = heapq.heappop(fringe)
            if curr[1]==destinationindex:
                return self.make_bfs_path_list(prev,curr[1],bot)
            #adjlist = get_adj_indices(curr[1][0],curr[1][1],len(grid))
            if curr in newweights.keys():
                if not newweights[curr]==None:
                    for child in newweights[curr]:
                        if grid[child[1]]%10==0 or grid[child[1]]%10==2:
                            temp_dist = distances[curr]+child[0]
                            if (not child in distances) or temp_dist<distances[child]:
                                distances[child]=temp_dist
                                prev[child[1]]=curr[1]
                                heapq.heappush(fringe,child)
        
        return []
    def bot_3_dynamic_UFCS_2mice(self,grid,start,destinationindex,state,bot,shortdist,evidence,t):
        #exactly the same as other UFCS, just with two mice
        #turn 3d UFCS into 2d by making weights list of arrays into a hashmap/graph that UFCS algorithm can traverse
        newweights = {(0,start,0):None}
        seen = []
        currlevel = [(0,start,0)]
        j=0
        rowsign = destinationindex[0]-start[0]
        colsign = destinationindex[1]-start[1]
        check = False
        tempstate = state.copy()
        while not currlevel==[]:
            predictedmargstate,tempstate = self.predicting_2mice(tempstate.copy(),shortdist,evidence.copy(),grid)
            nextlevel = []
            for item in currlevel:
                newweights[item] = None
                if item[1] == destinationindex:
                    check = True
                    break
                adjsqrs = self.get_adj_indices(item[1][0],item[1][1],len(grid))
                for sq in adjsqrs:
                    if (grid[sq]%10==0 or grid[sq]%10==2) and not sq in seen:
                        #These next four lines are made so that if the destination is up and to the right of the start for instance,
                        #the UFCS graph can only point up or right
                        if (sq[0]-item[1][0]==0 or (sq[0]-item[1][0]>0 and rowsign>0) or (sq[0]-item[1][0]<0 and rowsign <0)):
                            if (sq[1]-item[1][1]==0 or (sq[1]-item[1][1]>0 and colsign>0) or (sq[1]-item[1][1]<0 and colsign<0)):
                                if min(start[0],destinationindex[0])<=sq[0]<=max(start[0],destinationindex[0]):
                                    if min(start[1],destinationindex[1])<=sq[1]<=max(start[1],destinationindex[1]):
                                        new = (1+math.log(1-predictedmargstate[sq]),sq,item[2]+1)
                                        if not new in nextlevel:
                                            nextlevel.append(new)
                                        if newweights[item]==None:
                                            newweights[item]=[new]
                                        else:
                                            newweights[item].append(new)
                seen.append(item[1])
            if check == True:
                break
            currlevel = nextlevel
            j+=1
        check = False
        for item in list(newweights.values()):
            if not item == None:
                for item2 in item:
                    if item2[1]==destinationindex:
                        check = True
        if not check:
            newweights = {(0,start,0):None}
            seen = []
            currlevel = [(0,start,0)]
            j=0
            check = False
            tempstate = state.copy()
            while not currlevel==[]:
                predictedmargstate,tempstate = self.predicting_2mice(tempstate.copy(),shortdist,evidence.copy(),grid)
                nextlevel = []
                for item in currlevel:
                    newweights[item] = None
                    if item[1] == destinationindex:
                        check = True
                        break
                    adjsqrs = self.get_adj_indices(item[1][0],item[1][1],len(grid))
                    for sq in adjsqrs:
                        if (grid[sq]%10==0 or grid[sq]%10==2) and not sq in seen:
                            new = (1+math.log(1-predictedmargstate[sq]),sq,item[2]+1)
                            if not new in nextlevel:
                                nextlevel.append(new)
                            if newweights[item]==None:
                                newweights[item]=[new]
                            else:
                                newweights[item].append(new)
                    seen.append(item[1])
                if check == True:
                    break
                currlevel = nextlevel
                j+=1
        distances = {}
        prev = {}
        fringe = []
        distances[(0,start,0)]=0
        prev[start]= None
        heapq.heappush(fringe,(0,start,0))
        while not fringe == []:
            curr = heapq.heappop(fringe)
            if curr[1]==destinationindex:
                return self.make_bfs_path_list(prev,curr[1],bot)
            #adjlist = get_adj_indices(curr[1][0],curr[1][1],len(grid))
            if curr in newweights.keys():
                if not newweights[curr]==None:
                    for child in newweights[curr]:
                        if grid[child[1]]%10==0 or grid[child[1]]%10==2:
                            temp_dist = distances[curr]+child[0]
                            if (not child in distances) or temp_dist<distances[child]:
                                distances[child]=temp_dist
                                prev[child[1]]=curr[1]
                                heapq.heappush(fringe,child)
        
        return []


    def bfs(self,grid,start,end,bot):
        #searches grid for end index
        prev = {}
        marked = []
        queue = deque()
        queue.append(start)
        prev[start] = None
        marked.append(start)
        while not len(queue)==0:
            currentstate = queue.popleft()
            if currentstate == end:
                return self.make_bfs_path_list(prev,currentstate,bot)
            adjindexlist = self.get_adj_indices(currentstate[0],currentstate[1],len(grid))
            for item in adjindexlist:
                if (grid[item]%10==0 or grid[item]%10==2) and item not in marked:
                    queue.append(item)
                    marked.append(item)
                    prev[item] = currentstate
        return []
    def make_bfs_path_list(self,prev,currentstate,bot):
        #An extension of the bfs function, just makes the actual list of indices/tuples to return
        bfslist = []
        prevcopy = prev.copy()
        currentstatecopy = currentstate
        while not prev[currentstate]==None:
            #print('Current state: ',currentstate)
            bfslist.append(currentstate)
            if bot==2 or bot==3:
                bfslist.append(None)
                #Nones are so bots 2 and 3 can alternate moving and sensing
            currentstate = prev[currentstate]
        if bot==2 or bot==3:
            bfslist.pop()
        bfslist.reverse()
        if bot==3:
            while len(bfslist)>9:
                bfslist.pop()
        return bfslist

    def create_grid(self,D):
        #create array
        newgrid = np.ones((D,D))
        #select random block to open
        row = math.floor((random.random())*D)
        col = math.floor((random.random())*D)
        newgrid[row][col] = 0
        oneneighborlist = self.get_adj_indices(row,col,D)
        #iteratively open new blocks
        while not oneneighborlist==[]:
            #Randomly select from one-neighbor list to open
            listsize = len(oneneighborlist)
            listchoice = math.floor(listsize*random.random())
            newgrid[oneneighborlist[listchoice]] = 0
            #Iterate through grid to find all cells with exactly one open neighbor
            oneneighborlist = []
            for i in range(D):
                for j in range(D):
                    if newgrid[i][j]==1:
                        adjlist = self.get_adj_indices(i,j,D)
                        openneighborcount = 0
                        for item in adjlist:
                            if newgrid[item]==0:
                                openneighborcount += 1
                        if openneighborcount == 1:
                            oneneighborlist.append((i,j))
        #Get list of open cells with one open neighbor
        deadends = []
        for i in range(D):
            for j in range(D):
                if newgrid[i][j]==0:
                    adjlist = self.get_adj_indices(i,j,D)
                    openneighborcount = 0
                    for item in adjlist:
                        if newgrid[item]==0:
                            openneighborcount += 1
                    if openneighborcount == 1:
                        deadends.append((i,j))
        #Get list of closed neighbors of deadend cells
        closednbrsofdeadends = []
        for item in deadends:
            adjlist = self.get_adj_indices(item[0],item[1],D)
            for item2 in adjlist:
                if newgrid[item2] == 1:
                    closednbrsofdeadends.append(item2)
        #Open approximately half of these closed neighbors of deadend cells
        #Well, a bit more than half, it seemed like this gave greater variance in success rates and thus made things a little bit more interesting
        #i.e. more corridors to choose from, more significant choices for the bots to make
        openeddeadends = []
        for item in closednbrsofdeadends:
            if random.random() < .5:
                newgrid[item] = 0
                openeddeadends.append(item)
        #print(newgrid)
        #print('Before: ',len(closednbrsofdeadends),' After: ', len(closednbrsofdeadends)-len(openeddeadends))
        return newgrid
    def place_initial_positions(self,grid,nummice):
        opencelllist=[]
        for i in range(len(grid)):
            for j in range(len(grid)):
                if grid[(i,j)]==0:
                    opencelllist.append((i,j))
        botindex = opencelllist[math.floor(random.random()*len(opencelllist))]
        grid[botindex] = 1110
        opencelllist.remove(botindex)
        mouseindex1 = opencelllist[math.floor(random.random()*len(opencelllist))]
        grid[mouseindex1] = 2
        if nummice == 2:
            opencelllist.remove(mouseindex1)
            mouseindex2 = opencelllist[math.floor(random.random()*len(opencelllist))]
            grid[mouseindex2] = 2
            return grid,botindex,mouseindex1,mouseindex2
        else:
            return grid,botindex,mouseindex1,None
    def get_adj_indices(self,row,col,D):
        #Citation: I got the idea of this from https://stackoverflow.com/questions/51657128/how-to-access-the-adjacent-cells-of-each-elements-of-matrix-in-python
        #I use this function very frequently
        #Just returns indices and accounts for borders
        adjindlist = []
        if row+1<D:
            adjindlist.append((row+1,col))
        if row>0:
            adjindlist.append((row-1,col))
        if col+1<D:
            adjindlist.append((row,col+1))
        if col>0:
            adjindlist.append((row,col-1))
        return adjindlist
    def calc_manhattan_dist(self,tup1,tup2):
        return int(math.fabs(tup1[0]-tup2[0])+math.fabs(tup1[1]-tup2[1]))

    def run_sim_2mice_bot1only(self,grid,a,nummice,stoch,printyn):
        grid,botindex,mouseindex1,mouseindex2 = self.place_initial_positions(grid,nummice)
        if printyn:
            file = open('shipresults.txt','a')
            file.write('t= 0 \n')
            file.write(np.array2string(grid,max_line_width=1000))
            file.write('\n')
            file.close()
        bot1index,bot2index,bot3index = botindex,botindex,botindex
        done = False
        currentstate,openpairslist = self.get_initial_dist_2mice(grid)
        bot1state,bot2state,bot3state = currentstate.copy(),currentstate.copy(),currentstate.copy()
        bot1evidence,bot2evidence,bot3evidence = [None],[None],[None]
        #Evidence format: (t,type,tuple of index on ship) 
        #where type = 0 if negative sense, 1 if positive sense, 2 if walked into square"""
        bot1plan,bot2plan,bot3plan = [],[],[]
        bot1mode,bot2mode,bot3mode = 2,2,2
        t = 1
        runsimresults = [None,None,None,None,None,None,a]
        while bot1mode>0:
            if bot1mode==2:
                grid,bot1index,bot1evidence,bot1plan,bot1state,bot1margstate = self.bot_1_2mice(grid,bot1index,t,bot1plan,bot1evidence.copy(),mouseindex1,mouseindex2,a,bot1state.copy(),stoch,openpairslist)
            if bot1mode==1:
                if b1secondmouse == 1:
                    grid,bot1index,bot1evidence,bot1plan,bot1state1mouse = self.bot_1(grid,bot1index,t,bot1plan,bot1evidence.copy(),mouseindex1,a,bot1state1mouse.copy(),stoch)
                if b1secondmouse == 2:
                    grid,bot1index,bot1evidence,bot1plan,bot1state1mouse = self.bot_1(grid,bot1index,t,bot1plan,bot1evidence.copy(),mouseindex2,a,bot1state1mouse.copy(),stoch)
            if stoch:
                grid,mouseindex1,mouseindex2 = self.move_mice(grid,mouseindex1,mouseindex2,bot1mode)
            if bot1mode==2 and (bot1index==mouseindex1 or bot1index==mouseindex2):
                bot1mode=1
                if bot1index==mouseindex1:
                    bot1mode=1
                    b1secondmouse=2
                    grid[bot1index]+=10
                    bot1oldstate = currentstate.copy()

                    print(np.array2string(self.calc_marginal_dists(bot1oldstate,openpairslist,len(grid)),max_line_width=1000))
                    for k in range(1,t):
                        if not bot1evidence[k][2]==mouseindex1:
                            bot1oldstate = self.filtering_2mice(bot1oldstate.copy(),stoch,bot1evidence.copy(),k,grid.copy(),a,openpairslist)
                            print(bot1evidence[k])
                            print(np.array2string(self.calc_marginal_dists(bot1oldstate,openpairslist,len(grid)),max_line_width=1000))
                            k+=1
                    bot1state1mouse = self.get_marginal_dist_for_index(bot1oldstate.copy(),mouseindex1,openpairslist,len(grid),grid.copy())
                    #print(np.array2string(bot1state1mouse,max_line_width=1000))
                if bot1index==mouseindex2:
                    bot1mode=1
                    b1secondmouse=1
                    grid[bot1index]+=10
                    bot1oldstate = currentstate.copy()
                    k=1
                    print(np.array2string(self.calc_marginal_dists(bot1oldstate,openpairslist,len(grid)),max_line_width=1000))
                    for k in range(1,t):
                        if not bot1evidence[k][2]==mouseindex2:
                            bot1oldstate = self.filtering_2mice(bot1oldstate.copy(),stoch,bot1evidence.copy(),k,grid.copy(),a,openpairslist)
                            print(bot1evidence[k])
                            print(np.array2string(self.calc_marginal_dists(bot1oldstate,openpairslist,len(grid)),max_line_width=1000))
                    bot1state1mouse = self.get_marginal_dist_for_index(bot1oldstate.copy(),mouseindex1,openpairslist,len(grid),grid.copy())
                runsimresults[0]=t
                bot1plan = []

            if bot1mode==1 and ((b1secondmouse==1 and bot1index==mouseindex1) or (b1secondmouse==2 and bot1index==mouseindex2)):
                grid[bot1index]-=10
                if b1secondmouse==1:
                    grid[mouseindex2]-=10
                else:
                    grid[mouseindex1]-=10
                runsimresults[1] = t
                bot1mode = 0
            #if stoch:
            #   grid,mouseindex1,mouseindex2 = move_mice(grid,mouseindex)
            if printyn:
                file = open('shipresults.txt','a')
                file.write('t= ')
                file.write(str(t))
                file.write('\n')
                file.write(np.array2string(grid, max_line_width=1000))
                file.write('\n')
            if printyn:
                if bot1mode>0:
                    file.write('\n Bot 1 recent evidence: ')
                    file.write(str(bot1evidence[t]))
                    #file.write('\n Bot 1 full evidence: ')
                    #file.write(str(bot1evidence))
                    file.write('\n Bot 1 estimated state: \n')
                    if bot1mode==2:
                        file.write(np.array2string(bot1margstate,max_line_width=1000))
                    if bot1mode==1:
                        file.write(np.array2string(bot1state1mouse,max_line_width=1000))
                    file.write('\n')
            t += 1
        return runsimresults
    
    def get_marginal_dist_for_index(self,state,margindex,openpairslist,D,grid,check):
        return self.calc_marginal_dists(state,openpairslist,len(grid))
