import numpy as np
import logging


class TestingSuite():
    def __init__(self, humans, robot, config):
        self.scenarios = list()
        self.testing_scenario = {
                                    'goal_change': self.change_goal,
                                    'adversarial':self.adversarial_agent,
                                    'adversarial_by_birth': self.adversarial_by_birth,
                                    'kidnapped_robot_problem': 4,
                                    'Static_Obstacles': True,
                                    'sudden_speed_up':self.sudden_speed_up
                                    }
        
        self.config = config
        
        self.debug = False
        try:
            self.conflict_radius = config.getfloat("testing-suite", "extended_parameter")
            self.epsilon = config.getfloat("testing-suite", "epsilon")
            scenarios = config.get("testing-suite","scenarios")
            self.adversarial_agent_type = config.get("testing-suite","adversarial_agent_type")
            lst = scenarios.split(' ')
            for elm in lst:
                if elm == 'NA':
                    continue
                elif not elm in self.testing_scenario.keys():
                    logging.info(f'The {elm} is not a valid testing scenario')
                    continue
                
                self.scenarios.append(elm)
            debug = config.get("testing-suite","debug")
            if debug == 'true':
                self.debug = True 
            elif debug == 'false':
                self.debug = False
        except:
            raise Exception(f'An error has occured')
            pass
        
        # logging.info("The testing scenarios are: " + " ".join(self.scenarios))
        
        self.square_width = config.getfloat('sim', 'square_width')
        self.circle_radius = config.getfloat('sim', 'circle_radius')
        self.robot = robot
        self.humans = humans
        self.human_num = len(humans)
        self.adversarial_agent_list = np.zeros(self.human_num)
        if 'adversarial_by_birth' in self.scenarios:
            self.adversarial_by_birth_agent = np.random.randint(1,self.human_num)
        else:
            self.adversarial_by_birth_agent = None
        if self.debug and self.adversarial_by_birth_agent:
            logging.info(f"The adverserial agent by birth is {self.adversarial_by_birth_agent} and the agent type is {self.adversarial_agent_type}")
        self.speed_up_list =  np.zeros(self.human_num)
        
    def testing_suite(self, human, human_id = None)->None:
        '''
        The testing suite method incorporates all the crowd modelling scenarios implemented on one human
        '''
        if not len(self.scenarios):
            return 
        if not self.human_num:
            # Return the method if the number of humans are zero
            return  
        scenarios = self.scenarios
        if human_id == None:
            raise  Exception("Please input the id of the human") 
        radius_check = self.conflict_cirlce_check(human)
        
        for scenario in scenarios:
      
            if self.testing_scenario[scenario] == self.testing_scenario['adversarial_by_birth']:
                    self.testing_scenario[scenario]()
            elif radius_check:
                if self.testing_scenario[scenario] == self.testing_scenario['sudden_speed_up']:
                    self.testing_scenario[scenario](human_id)
                else:
                    self.testing_scenario[scenario](human,  human_id = human_id)
            
        return
    def conflict_cirlce_check(self,human)-> bool:
        '''
        This function checks the circle heuristic, i.e. if the human is in the conflict radius 
        Parameters:
        Conflict_Radius (int): The radius of the conflict cirlce
        Returns:
        check (bool):   True if human is in the circle
                        False if human is outside the circle
        '''
        human_pose = (human.px, human.py)
        robot_pose = (self.robot.px, self.robot.py)

        # Check if the i_th human is in the 
        check =  (human_pose[0] -robot_pose[0])**2 +(human_pose[1] -robot_pose[1])**2 <= self.conflict_radius**2    
        return check
    def change_goal(self, human, epsilon = None, human_id = None):
        if epsilon == None:
            epsilon = self.epsilon 
        old_goal = (human.gx, human.gy)
        angle = np.random.random()
        radius = np.random.uniform(0, self.circle_radius )
        random_variable = np.random.rand()
        
        if self.epsilon > random_variable:
            # print(f"This is {(random_variable,epsilon)}")
            gx = radius * np.cos(angle)
            gy = radius * np.sin(angle)
            human.gx = np.random.choice([1,-1])*gx
            human.gy = np.random.choice([1,-1])*gy
            # print(f'The goal position of the {human_id}-th human has been changed from {old_goal} to {(human.gx, human.gy)}')
        return
    
    def sudden_speed_up(self, human_id, epsilon = None):
        if epsilon == None:
            epsilon = self.epsilon 
        random_variable = np.random.rand()
        if not self.speed_up_list[human_id]:
            if epsilon > random_variable:
                self.speed_up_list[human_id] = 1
                # This is not idead since this will be a continual increase
                self.humans[human_id].v_pref = 5*self.humans[human_id].v_pref 
                if self.debug:
                    logging.info(f"The speed of human {human_id} has been increased to {self.humans[human_id].v_pref}")
        count = np.sum(self.speed_up_list)
        return
    def adversarial_agent(self, human, epsilon = None, human_id = None):
        if epsilon == None:
            epsilon = self.epsilon 
        random_variable = np.random.rand()
        if epsilon > random_variable:
            self.adversarial_agent_list[human_id] = 1
        count = 0
        for i in range(len(self.adversarial_agent_list)):
            if self.adversarial_agent_list[i]:
                self.humans[i].gx = self.robot.px
                self.humans[i].gy = self.robot.py
                count += 1
        # print(f'The number of adversarial agents are {np.sum(self.adversarial_agent_list)}')
        return 
    def sudden_change_of_intent(self, epsilon = None ):
        if epsilon == None:
            epsilon = self.epsilon 
        if self.human_num == 0:
            return None
        else: 
            angle = np.random.random()
            stochastic_arr = np.random.rand(self.human_num)
            stochastic_arr[stochastic_arr > epsilon]
            unique,counts = np.unique(stochastic_arr, return_counts= True)
            counts_dic = dict(zip(unique,counts))
            sorter = np.argsort(stochastic_arr)
            indexes = sorter[np.searchsorted(stochastic_arr,unique, sorter= sorter)]
            for i in indexes:
                gx = self.circle_radius * np.cos(angle)
                gy = self.circle_radius * np.cos(angle)
                goal = (gx, gy)
                self.humans[i].gx = np.random.choice([1,-1])*gx
                self.humans[i].gy = np.random.choice([1,-1])*gy
                # print(f"The human {i} just changed their goal to {goal}")
    def adversarial_by_birth(self):
        #Maybe this method shouldnt be added in the testing suite since testing suite is called in a loop over 
        '''
        There is an error: THe ORCA agent just stops when it gets close 
        Two types:
            1. Fast
            2. Slow 
        '''
        self.humans[self.adversarial_by_birth_agent].gx = self.robot.px
        self.humans[self.adversarial_by_birth_agent].gy = self.robot.py
        if self.adversarial_agent_type == 'fast':
            self.sudden_speed_up(self.adversarial_by_birth_agent, epsilon = 1) #This will ensure that the adversarial agent has a greater speed 

        return 
