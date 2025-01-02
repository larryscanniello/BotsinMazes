
import numpy as np
import random
import math
import heapq
from collections import deque
from collections import Counter

class FireBots:
    def __init__(self):
        pass

    def main(self):
        #Main function. Just setting up menu options, passing result to run_fire_interface
        menunum = -1
        np.set_printoptions(suppress=True)
        while not menunum == 0:
            menunum = int(input("""\nSelect mode:
1) Random flammability per simulation, random layout per simulation
2) Fixed flammability per simulation, random layout per simulation
3) Fixed flammability per simulation, random layout fixed for all simulations
4) Overwrite/make blank old shipresults file
0) Exit\n"""))
            if menunum == 1 or menunum == 2 or menunum == 3 or menunum==5:
                self.run_fire_interface(menunum)
        
            if menunum == 4:
                file = open('shipresults.txt','w')
                file.write(' ')
                file.close()

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
            if random.random() > .62:
                newgrid[item] = 0
                openeddeadends.append(item)
        #print(newgrid)
        #print('Before: ',len(closednbrsofdeadends),' After: ', len(closednbrsofdeadends)-len(openeddeadends))
        return newgrid

    def run_fire_interface(self,menunum):
        #Just a continuation of the menu interface. Different options for different values of menunum.
        #This is where the main looping and counting of success happens when performing multiple iterations of simulations
        #b1,b2,b3,b4,solpossible are boolean variables returned that indicate success of specific simulations
        #Every 25 iterations/simulations, prints 'x simulations done'
        D = int(input('\nTo make DxD layouts, enter D (between 20 and 30 recommended): \n'))
        iterations = int(input('\nEnter number of simulations to run: \n'))
        printyn = int(input("""\nResults will be printed to terminal at the end.                   
Enter 1 to visualize simulations and write all grids to shipresults.txt
Enter 2 to write simulations to shipresults.txt only when bot 4 fails and others succeed, or when all fail but solution possible
Enter 0 to write nothing to shipresults.txt (recommended if running many simulations)\n"""))
        bot1successes = 0
        bot2successes = 0
        bot3successes = 0
        bot4successes = 0 
        solpossibles = 0
        if menunum == 1:
            print('\nSimulations have begun.')
            for i in range(iterations):
                newgrid = self.create_grid(D)
                b1,b2,b3,b4,solpossible=self.run_fire(newgrid.copy(),random.random()**2.5,printyn,i)
                if b1==True:
                    bot1successes += 1
                if b2==True:
                    bot2successes += 1
                if b3==True:
                    bot3successes += 1
                if b4==True:
                    bot4successes += 1
                if solpossible==True:
                    solpossibles += 1
                if i%25==0 and i>0:
                    print(i,' simulations done')
        if menunum == 2:
            q = float(input('\nEnter q, the flammability parameter, where 0<q<1. Higher q = more flammable ship.\n'))
            print('\nSimulations have begun.')
            for i in range(iterations):
                newgrid = self.create_grid(D)
                b1,b2,b3,b4,solpossible=self.run_fire(newgrid.copy(),q,printyn,i)
                if b1==True:
                    bot1successes += 1
                if b2==True:
                    bot2successes += 1
                if b3==True:
                    bot3successes += 1
                if b4==True:
                    bot4successes += 1
                if solpossible==True:
                    solpossibles += 1
                if i%25==0 and i>0:
                    print(i,' simulations done')
        if menunum == 3:
            q = float(input('\nEnter q, the flammability parameter, where 0<q<1. Higher q = more flammable ship.\n'))
            newgrid = self.create_grid(D)
            print('\nSimulations have begun.')
            for i in range(iterations):
                b1,b2,b3,b4,solpossible=self.run_fire(newgrid.copy(),q,printyn,i)
                if b1==True:
                    bot1successes += 1
                if b2==True:
                    bot2successes += 1
                if b3==True:
                    bot3successes += 1
                if b4==True:
                    bot4successes += 1
                if solpossible==True:
                    solpossibles += 1
                if i%25==0 and i>0:
                    print(i,' simulations done')
                    
        print('Bot 1 successes: ',bot1successes)
        print('Bot 2 successes: ',bot2successes)
        print('Bot 3 successes: ',bot3successes)
        print('Bot 4 successes: ',bot4successes)
        print('Success possible: ',solpossibles)
        
    def run_fire(self,grid,q,printyn,p):
        #run_fire is run once per main simulation
        #This next line of code sends grid of 0's and 1's, which is passed in, to place_initial_positions, where button, bots, and fire is placed
        grid,botindex,extindex,fireindex = self.place_initial_positions(grid,printyn)
        bot1index = botindex
        bot2index = botindex
        bot3index = botindex
        bot4index = botindex
        #Get path for bot 1, bot 1 will not waver from this path
        bot1bfs = self.bfs(grid,bot1index)
        t = 1
        #make a list to store grids as they evolve at t=0,1,2,...n so the simulation can be examined/printed retroactively
        arrlist = []
        arrlist.append(grid.copy())
        #find switch for bot4
        for i in range(len(grid)):
            for j in range(len(grid)):
                if grid[(i,j)]%10==5:
                    switch = (i,j)
        #bot 4 initial route is planned, this is explained in detail in my write-up
        bot4weights = self.get_bot_4_weights(grid,bot4index,switch,q)
        bot4route = self.bot_4_UFCS(grid,bot4index,bot4weights)
        #safeguard is the check for bot 4 to make sure its path is on fire. is safeguard==True, then bot 4 is in bot 3 mode
        safeguard = False
        while self.check_done(grid)==False:
            #the grid is passed back and forth between functions for each bot movement/fire step
            if not bot1bfs == []:
                grid,bot1index = self.move_bot_1(grid,bot1index,bot1bfs)
            grid,bot2index = self.move_bot_2(grid,bot2index)
            grid,bot3index = self.move_bot_3(grid,bot3index)
            grid,bot4index,safeguard = self.move_bot_4(grid,bot4index,bot4route,safeguard,q,p,t)
            grid = self.fire_step(grid,q)
            #If option is selected, prints grids as they evolve
            if printyn == 1:
                self.print2shipresults(grid,t)
            arrlist.append(grid.copy())
            t+=1
        #At end of main simulation, check if bots have succeeded
        if grid[bot1index]%10==5:
            b1 = True
        else: b1= False
        if grid[bot2index]%10==5:
            b2 = True
        else: b2 = False
        if grid[bot3index]%10==5:
            b3 = True
        else: b3 = False
        if grid[bot4index]%10==5:
            b4 = True
        else: b4 = False

        #At end of main simulation, if each bot failed, check in hindsight if a solution was possible
        if b1==False and b2==False and b3==False and b4==False:
            solpossible,sollist = self.is_possible_sol(arrlist)
        else: solpossible = True
        #If option selected, print hindsight path if there is one
        if printyn==2 and (b1==False and b2==False and b3==False and b4==False and solpossible==True):
            for i in range(1,len(arrlist)):
                if i-1<len(sollist):
                    arrlist[i][sollist[i-1]]+=200
            file = open('shipresults.txt','a')
            t=0
            file.write('p= ')
            file.write(str(p))
            file.write('\n')
            for item in arrlist:
                file.write('t= ')
                file.write(str(t))
                file.write('\n')
                file.write(np.array2string(item,max_line_width=1000))
                file.write('\n')
                t += 1
        #If option selected, print grids if bot 4 failed and another bot succeeded
        if (b1==True or b2==True or b3==True) and b4==False and printyn==2:
            file = open('shipresults.txt','a')
            t=0
            file.write('p= ')
            file.write(str(p))
            file.write('^\n')
            for item in arrlist:
                file.write('t= ')
                file.write(str(t))
                file.write('\n')
                file.write(np.array2string(item,max_line_width=1000))
                file.write('\n')
                t += 1
        return b1,b2,b3,b4,solpossible
        
    def move_bot_1(self,grid,bot1index,bot1bfs):
        #If bot 1 is on the button or fire, return without moving. Same thing for other bots
        if grid[bot1index]%10==5 or grid[bot1index]%10==2:
            return grid,bot1index
        try: 
            index = bot1bfs.index(bot1index)
        except:
            print('Error: move_bot_1 ')
        grid[bot1index] -= 10
        grid[bot1bfs[index+1]] += 10
        return grid,bot1bfs[index+1]
    def move_bot_2(self,grid,bot2index):
        if grid[bot2index]%10==2 or grid[bot2index]%10==5:
            return grid,bot2index
        #bot 2 does bfs at every step
        bfslist = self.bfs(grid,bot2index)
        if bfslist == []:
            return grid,bot2index
        grid[bfslist[0]] -= 100
        grid[bfslist[1]] += 100
        bot2index = bfslist[1]
        return grid,bot2index
    def move_bot_3(self,grid,bot3index):
        if grid[bot3index]%10==2 or grid[bot3index]%10==5:
            return grid,bot3index
        #change all open cells with fire neighbors to 3
        altgrid = grid.copy()
        for i in range(len(altgrid)):
            for j in range(len(altgrid)):
                if grid[(i,j)]%10==0:
                    adjlist = self.get_adj_indices(i,j,len(altgrid))
                    checkfireneighbor = False
                    for item in adjlist:
                        if grid[item]%10==2:
                            checkfireneighbor = True
                            altgrid[(i,j)]=3
                            break
        #do bfs on alt grid, if that doesn't work, do bfs on normal grid                
        bfslist = self.bfs(altgrid,bot3index)
        if not bfslist == []:
            grid[bot3index] -= 1000
            grid[bfslist[1]] += 1000
            return grid,bfslist[1]
        else:
            bfslist = self.bfs(grid,bot3index)
            if not bfslist==[]:
                grid[bot3index] -= 1000
                grid[bfslist[1]] += 1000
                return grid,bfslist[1]
            else:
                return grid,bot3index
        return
    def move_bot_4(self,grid,bot4index,bot4route,safeguard,q,p,t):
        if grid[bot4index]%10==2 or grid[bot4index]%10==5:
            return grid,bot4index,safeguard
        if bot4route==[]:
            return grid,bot4index,safeguard 
        #safeguard==True means bot 4 is in bot 3 mode  
        if safeguard:
            grid,bot4index = self.bot_4_does_bot_3(grid,bot4index)
            return grid,bot4index,safeguard
        else:
            #At every step, checks to see if bot 3 mode needed
            safeguard = self.bot_4_check_route(grid,bot4index,bot4route,q,p,t)
            if not safeguard:
                index = bot4route.index(bot4index)
                grid[bot4index]-=10000
                grid[bot4route[index+1]]+=10000
                return grid,bot4route[index+1],safeguard
            else:
                grid,bot4index = self.bot_4_does_bot_3(grid,bot4index)
                return grid,bot4index,safeguard
    def fire_step(self,grid,q):
        #pushes the fire along by a frame according to q
        newgrid = grid.copy()
        for i in range(len(grid)):
            for j in range(len(grid)):
                if grid[(i,j)]%10==0:
                    adjlist = self.get_adj_indices(i,j,len(grid))
                    neighborfirecount = 0
                    for item in adjlist:
                        if grid[item]%10 == 2:
                            neighborfirecount+=1
                    if random.random()<(1-(1-q)**neighborfirecount) and not newgrid[(i,j)]%10==2:
                        newgrid[(i,j)] += 2
        return newgrid

    def bot_4_check_route(self,grid,start,route,q,p,t):
        #For every index along the route, check adjacent tiles to see if they're on fire. If so, return True, else return False
        index = route.index(start)
        for i in range(index,len(route)):
            adjlist = self.get_adj_indices(route[i][0],route[i][1],len(grid))
            for item in adjlist:
                if grid[item]%10==2:
                    return True
        return False
    def bot_4_does_bot_3(self,grid,index):
        #Literally just the bot 3 code with very few adjustments, I could've done it in one function but whatever.
        altgrid = grid.copy()
        for i in range(len(altgrid)):
            for j in range(len(altgrid)):
                if grid[(i,j)]==0:
                    adjlist = self.get_adj_indices(i,j,len(altgrid))
                    checkfireneighbor = False
                    for item in adjlist:
                        if grid[item]==2:
                            checkfireneighbor = True
                            break
                    if checkfireneighbor==True:
                        altgrid[(i,j)]=3
        bfslist = self.bfs(altgrid,index)
        if not bfslist == []:
            grid[index] -= 10000
            grid[bfslist[1]] += 10000
            return grid,bfslist[1]
        else:
            bfslist = self.bfs(grid,index)
            if not bfslist==[]:
                grid[index] -= 10000
                grid[bfslist[1]] += 10000
                return grid,bfslist[1]
            else:
                return grid,index


    def bfs(self,grid,start):
        #Just a straightforward bfs that searches the grid for a shortest path from the start to the button
        prev = {}
        marked = []
        queue = deque()
        queue.append(start)
        prev[start] = None
        marked.append(start)
        while not len(queue)==0:
            currentstate = queue.popleft()
            if grid[currentstate]%10==5:
                bfslist = self.make_bfs_path_list(prev,currentstate)
                return bfslist
            adjindexlist = self.get_adj_indices(currentstate[0],currentstate[1],len(grid))
            for item in adjindexlist:
                if (grid[item]%10==0 or grid[item]%10==5) and item not in marked:
                    queue.append(item)
                    marked.append(item)
                    prev[item] = currentstate
        return []
    def make_bfs_path_list(self,prev,currentstate):
        #An extension of the bfs function, just makes the actual list of indices/tuples to return
        bfslist = []
        while not prev[currentstate] == None:
            #print('Current state: ',currentstate)
            bfslist.append(currentstate)
            currentstate = prev[currentstate]
        bfslist.append(currentstate)
        bfslist.reverse()
        return bfslist

    def place_initial_positions(self,grid,printyn):
        D=len(grid)
        #Goal: Place bots, extinguisher, and initial fire cell
        #First step: Create list of all open cells
        opencelllist=[]
        for i in range(D):
            for j in range(D):
                if grid[i][j]==0:
                    opencelllist.append((i,j))
        #Randomly pick an open cell to place bots
        botindex = opencelllist[math.floor(random.random()*len(opencelllist))]
        grid[botindex] = 11110
        opencelllist.remove(botindex)
        #Randomly pick an open cell to place extinguisher
        extindex = opencelllist[math.floor(random.random()*len(opencelllist))]
        grid[extindex] = 5
        opencelllist.remove(extindex)
        #Randomly pick an open cell to start fire
        fireindex = opencelllist[math.floor(random.random()*len(opencelllist))]
        grid[fireindex] = 2
        opencelllist.remove(fireindex)
        if printyn == 1:
            file = open('shipresults.txt','a')
            file.write('t= 0 \n')
            file.write(np.array2string(grid, max_line_width=1000))
            file.write('\n')
            file.close()
        return grid,botindex,extindex,fireindex

    def check_done(self,grid):
        #Iterates through the entire grid, if we can find an open square with a bot on it then we are not done
        #In hindsight, more efficient to check at each bot index, but this works well enough
        for i in range(len(grid)):
            for j in range(len(grid)):
                if grid[i][j]%10==0 and not grid[i][j]==0:
                    return False
        return True    
    def check_done_b4fire(self,grid,switch,bot4index):
        #I am 99% sure sure this is not being used at all, and was used in an older iteration of the code, so just ignore this; keeping to be safe
        adjlist = self.get_adj_indices(switch[0],switch[1],len(grid))
        check1 = False
        for item in adjlist:
            if grid[item]%10==0:
                check1 = True
        if check1 == False:
            return True
        check = False
        for item in adjlist:
            if grid[item]%10==2:
                check = True
        if check == False:
            return False
        else: 
            newadjlist = []
            for item in adjlist:
                if grid[item]%10==0:
                    newadjlist.append(item)
            if not self.checkb4bfs(grid,switch):
                return True
            else: return False
            

        return True
    def checkb4bfs(self,grid,start):
        #Also 99% sure this isn't being used at all, keeping to be safe
        marked = []
        queue = deque()
        queue.append(start)
        marked.append(start)
        while not len(queue)==0:
            currentstate = queue.popleft()
            adjindexlist = self.get_adj_indices(currentstate[0],currentstate[1],len(grid))
            for item in adjindexlist:
                if grid[item]%10==0 and item not in marked:
                    queue.append(item)
                    marked.append(item)
                if grid[item]%10==2:
                    return True
        return False
    def is_possible_sol(self,arrlist):
        #arrlist is a list of arrays that make up a full main simulation
        #The purpose of this function is to check in hindsight after a main simulation is over to see if there was a solution
        #find start index in the starting array
        check = False
        for i in range(len(arrlist[0])):
            for j in range(len(arrlist[0])):
                if arrlist[0][(i,j)]>5:
                    start = (i,j)
                    check = True
                    break
            if check == True:
                break
        #Assumption: If there is a path that works, then there is a path that works with no backtracking
        #This is a BFS style search. Nodes from the previous level explored are in prev, their adjacent neighbors that we are currently exploring
        #are in curr. Then prev becomes curr, and we iterate until we have explored the sequence of grids
        #Confusingly, there is also previndex, so that when we do find a path, we can print this hindsight path; the function returns a list of indices
        #Also notice we look for arrlist[i][item2]%10==0, so that we are looking through the grids in sequence
        marked = [start]
        prev = [start]
        previndex = {start: None}
        for i in range(1,len(arrlist)):
            curr = []
            for item in prev:
                adjind = self.get_adj_indices(item[0],item[1],len(arrlist[0]))
                for item2 in adjind:
                    if item2 not in marked and arrlist[i][item2]%10==0:
                        curr.append(item2)
                        marked.append(item2)
                        previndex[item2] = item
                    if arrlist[0][item2]==5:
                        previndex[item2] = item
                        index = item2
                        pathlist = []
                        while not previndex[index]==None:
                            pathlist.append(index)
                            index = previndex[index]
                        pathlist.reverse()
                        return True,pathlist
                prev = curr
        return False,[]

    def get_bot_4_weights(self,grid,start,switch,q):
        #this is explained in detail in my write-up
        weightarrlist = []
        for i in range(50):
            gridcopy = grid.copy()
            firesequence = [gridcopy]
            for k in range(70):
                gridcopy = self.fire_step(gridcopy,q)
                firesequence.append(gridcopy)
            for i in range(len(firesequence)):
                if len(weightarrlist)>i:
                    weightarrlist[i][firesequence[i]%10==2]+=1
                else:
                    weightarrlist.append(np.ones((len(grid),len(grid))))
                    weightarrlist[i][firesequence[i]%10==2]+=1
        #uncomment this next part if you want it to print to a new file the actual weightarrlist
        """file2 = open('weightarrlist.txt','w')
        t=0
        for item in weightarrlist:
            file2.write('t=')
            file2.write(str(t))
            file2.write('\n')
            file2.write(np.array2string(item, max_line_width=1000))
            file2.write('\n')
            t+=1
        file2.close()
        """
        return weightarrlist
    def bot_4_UFCS(self,grid,start,weights):
        #turn 3d UFCS into 2d by making weights list of arrays into a hashmap/graph that UFCS algorithm can traverse
        #this entire function is also explained in detail in my write-up, so I will add no more comments
        newweights = {(0,start,0):None}
        seen = []
        currlevel = [(0,start,0)]
        j=0
        while j+1<len(weights) and not currlevel==[]:
            nextlevel = []
            for item in currlevel:
                newweights[item] = None
                adjsqrs =self.get_adj_indices(item[1][0],item[1][1],len(grid))
                for sq in adjsqrs:
                    if (grid[sq]%10==0 or grid[sq]%10==5) and not sq in seen:
                        new = (weights[item[2]+1][sq],sq,item[2]+1)
                        if not new in nextlevel:
                            nextlevel.append(new)
                        if newweights[item]==None:
                            newweights[item]=[new]
                        else:
                            newweights[item].append(new)
                seen.append(item[1])
            currlevel = nextlevel
            j+=1
        #now for the UFCS
        #print(len(weights))
        #print(newweights)
        distances = {}
        prev = {}
        fringe = []
        distances[(0,start,0)]=0
        prev[start]= None
        heapq.heappush(fringe,(0,start,0))
        while not fringe == []:
            curr = heapq.heappop(fringe)
            if grid[curr[1]]%10==5:
                return self.make_bfs_path_list(prev,curr[1])
            #adjlist = get_adj_indices(curr[1][0],curr[1][1],len(grid))
            try:
                if not newweights[curr]==None:
                    for child in newweights[curr]:
                        if grid[child[1]]%10==0 or grid[child[1]]%10==5:
                            temp_dist = distances[curr]+child[0]
                            if (not child in distances) or temp_dist<distances[child]:
                                distances[child]=temp_dist
                                prev[child[1]]=curr[1]
                                heapq.heappush(fringe,child)
            except:
                print('Error in Bot_4_UFCS')
        return []
    def bot_4_project_fire(self,grid,bot4index,bot4UFCS,q):
        #I believe this is also dead code used in an older iteration of my project
        gridcopy=grid.copy()
        index = bot4UFCS.index(bot4index)
        for i in range(index+1,len(bot4UFCS)):
            try:
                gridcopy[bot4UFCS[i-1]]-=10000
                gridcopy[bot4UFCS[i]]+=10000
            except:
                print(bot4UFCS)
            gridcopy = self.fire_step(gridcopy,q)
            if gridcopy[bot4UFCS[i]]%10==2:
                return True
        return False
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

    def print2shipresults(self,grid,t):
        file = open('shipresults.txt','a')
        file.write('t= ')
        file.write(str(t))
        file.write('\n')
        file.write(np.array2string(grid, max_line_width=1000))
        file.write('\n')
        file.close()

    if __name__ == '__main__':
        main()
        
