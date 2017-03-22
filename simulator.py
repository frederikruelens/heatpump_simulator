import numpy as np

class Flexible_load():

    def __init__(self, logger,  weather, tap, device='heatpump',  model_order=2, backup_method='discrete', n_actions=2, stochasticity=0):
        self.model_order = model_order
        self.device = device
        if device == 'heatpump':
            logger.debug('Created heat pump model with model order of ' + str(model_order) + ' and stochasticity level of ' + str(stochasticity) )
            if model_order == 1:
                self.model = RC_model(weather, backup_method, n_actions=n_actions)
            elif model_order == 2:
                self.model = RCRC_model(weather, backup_method, n_actions=n_actions, stochasticity=stochasticity)
            self.model.n_steps_quarter = 15

        elif device == 'boiler':
            self.model = boiler(tap, backup_method)

    def get_model(self):
        model = self.model
        return model

class RCRC_model():
    def __init__(self, weather, backup_method, Tair=21, Tmass=21 , Tmin=20, Tmax=23, power=2000, n_actions=2, stochasticity=0):
        self.Tair = Tair
        self.Tmass = Tmass
        self.T_min = Tmin
        self.T_max = Tmax
        self.power = power
        self.n_actions = n_actions
        self.action_space = np.linspace(0, self.power, self.n_actions)
        self.Ca = 2441000.0
        self.Ua = 125.0
        self.backup_method = backup_method
        self.coefficient_of_performance = 3
        self.deltaT = 60.0
        self.Cm = 9000000.0
        self.Hm = 6863.0
        self.model_order = 2
        self.device = 'heatpump'
        self.second = 0
        self.minute = 0
        self.quarter = 0
        self.episode = 0
        self.minute_day = 0
        self.quarter_day = 0
        self.flow_measurement = 0
        self.weather = weather
        self.stochasticity = stochasticity

    def set_clock(self):
        import numpy as np
        self.second = np.int(self.second + self.deltaT)
        self.minute = self.second / 60
        self.quarter = np.int(self.second / (60 * 15))
        self.episode = np.int(self.second / (60 * 1440))
        self.minute_day = np.mod(self.minute, 1440)
        self.quarter_day = np.mod(self.quarter, 96)

    def get_clock(self):
         print 'second ' + str(self.second)
         print 'minute ' + str(self.minute)
         print 'quarter ' + str(self.quarter)
         print 'episode ' + str(self.episode)
         print 'current minute of the day ' + str(self.minute_day)
         print 'current quarter of the day  ' + str(self.quarter_day)


    def print_info(self):
        print 'this is a (stochastic) battery model'

    def doSimStep(self, power, flow = 0):
        #self.get_clock()
        Tair = self.Tair
        Tmass = self.Tmass
        deltaT = self.deltaT

        Tout = self.weather.days_minute[self.episode, self.minute_day]
        Qhp = self.coefficient_of_performance*power
        self.Tair = Tair + deltaT/self.Ca * (Tmass*self.Hm - Tair*(self.Ua + self.Hm) +
                                             Qhp + Tout*self.Ua) + self.stochasticity*0.025*np.random.randn(1)
        self.Tmass = Tmass + deltaT / self.Cm * (self.Hm * (self.Tair - Tmass))

        self.set_clock()

    def backup_controller(self, action):
        type = self.backup_method
        self.upper_soc_bound = self.T_min
        self.lower_soc_bound = self.T_min
        if type == 'discrete':
            actual = action
            if self.Tair> self.T_max:
                 actual = 0
            elif self.Tair < self.T_min:
                 actual = self.power
        elif type == 'linear':
            self.upper_soc_bound = self.T_min * 1.025
            self.lower_soc_bound = self.T_min
            actual = action
            if self.Tair < self.upper_soc_bound:
                n_disc = 10
                bin = np.argmin(np.abs(np.linspace(self.lower_soc_bound, self.upper_soc_bound, n_disc) - self.Tair))
                heating_range = np.linspace(self.power, 0, n_disc)
                actual = max(heating_range[bin], action)
            if self.Tair > self.T_max:
                actual = 0
        return actual

    def get_temperatures(self):
        import numpy as np
        return np.array([1, 1])
    def get_soc(self):
        return 1
    def get_tap(self):
        return 1

    def get_Tout(self):
        return self.weather.days_minute[self.episode, self.minute_day]

'''
    -- lets test our simulation models
'''
if __name__ == '__main__':
    import numpy as np
    from weather import Weather
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from logger import Logger
    n_episodes = 1
    n_quarters = 96
    tuples = defaultdict(list)
    logger = Logger(name='simulator', show_in_console=True).create_logger()

    weather = Weather(season='winter')
    model = Flexible_load(logger, weather, 0, device='heatpump', model_order=2, backup_method='discrete',
                          n_actions=5, stochasticity=2).get_model()

    for episode in range(n_episodes):
        for quarter in range(n_quarters):
            print 'day ' + str(episode) + ' quarter ' + str(quarter)
            if np.random.rand(1) < 0.35:
                action = model.action_space[np.random.randint(0, model.n_actions, 1)]

            else:
                action = 0

            tuples['action'].append(action)
            tuples['Tair'].append(model.Tair)
            tuples['Tmass'].append(model.Tmass)
            tuples['Tout'].append(model.get_Tout())

            actual = model.backup_controller(action)
            for sim_steps in range(model.n_steps_quarter):
                model.doSimStep(power=actual)
            tuples['uphys'].append(actual)

    for key in tuples:
        tuples[key] = np.array(tuples[key])

    # %% ploteke
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(model.T_min * np.ones(np.shape(tuples['action'])))
    plt.plot(model.T_max * np.ones(np.shape(tuples['action'])))
    plt.plot(tuples['Tair'], label='temperature')
    plt.plot(tuples['Tmass'], label='temperature')
    #plt.plot(tuples['Tout'], label='temperature')
    plt.subplot(3, 1, 2)
    plt.plot(tuples['action'], label='uphys')
    plt.subplot(3, 1, 3)
    plt.plot(tuples['uphys'], label='action')
    plt.show()

    # mini diagnostic
    a = np.array(range(np.shape(tuples['action'])[0]))
    a = a.reshape(-1, 1)
    cutoff_temp = model.lower_soc_bound
    overrule_bump = a[(tuples['uphys'] != tuples['action']) & (tuples['Tair'] <= cutoff_temp)]
    overrule_indices = a[(tuples['uphys'] != 0) & (tuples['action'] == 0)]

    fig = plt.figure(2, figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tuples['Tair'], tuples['action'], tuples['uphys'], alpha=1, s=12, color='black')
    ax.scatter(tuples['Tair'][overrule_indices], tuples['action'][overrule_indices], tuples['uphys'][overrule_indices],
               s=50, color='red')
    ax.scatter(tuples['Tair'][overrule_bump], tuples['action'][overrule_bump], tuples['uphys'][overrule_bump],
               s=25, color='yellow')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Requested Action')
    ax.set_zlabel('Real Action')
    plt.show()
    if 1:
        n_overrules = np.float(np.shape(overrule_indices)[0])
        print 'slope percentage ' + str(n_overrules / np.shape(tuples['action'])[0])

        n_overrules = np.float(np.shape(overrule_bump)[0])
        print 'bump percentage ' + str(n_overrules / np.shape(tuples['action'])[0])
