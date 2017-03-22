class Weather(object):

    def __init__(self, season):
        import getpass
        import numpy as np
        import scipy.io
        from collections import namedtuple
        self.season = season

        if self.season == 'winter':
            import os
            file_location = os.path.join(os.path.dirname(__file__), 'data/weatherData.mat')
            mat = scipy.io.loadmat(file_location)

            self.days_minute = mat['WeatherData']['days'][0][0]
            days_minute_filterd = []

            minimum_temperature = 4
            maximum_temperature = 17
            for d in range(self.days_minute.shape[0]):
                T = self.days_minute[d, :]
                if (T.min() > minimum_temperature) and (T.max() < maximum_temperature):
                    days_minute_filterd.append(T)

            self.days_minute = np.array(days_minute_filterd)

        if self.season == '2daysrepeat':
            temperature_block = []
            temperature_block.append(np.ones((1, 1440))*15)
            day2 = np.ones((1, 1440))*15
            day2[0,800:1100] = 5
            temperature_block.append(day2)
            temperature_block = np.squeeze(np.array(temperature_block))

            self.days_minute = np.tile(temperature_block, (50, 1))

        if self.season == '1daysrepeat':
            temperature_block = []
            temperature_block.append(np.ones((1, 1440))*12)
            temperature_block = np.squeeze(np.array(temperature_block))
            self.days_minute = np.tile(temperature_block, (100, 1))

        if self.season == '5winterdays':
            import os
            file_location = os.path.join(os.path.dirname(__file__), 'data/weatherData.mat')
            mat = scipy.io.loadmat(file_location)

            self.days_minute = mat['WeatherData']['days'][0][0]
            days_minute_filterd = []

            minimum_temperature = 10
            maximum_temperature = 15
            for d in range(self.days_minute.shape[0]):
                T = self.days_minute[d, :]
                if (T.min() > minimum_temperature) and (T.max() < maximum_temperature):
                    days_minute_filterd.append(T)

            days_minute_filterd = np.array(days_minute_filterd)
            self.days_minute = np.tile(days_minute_filterd, (20, 1))

        self.year_minute = self.days_minute.ravel()
        self.year_quarter = np.mean(self.year_minute.T.reshape(-1, 15), axis=1)
        self.days_quarter = self.year_quarter.reshape(-1, 96)

    def plot(self, days):
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.subplot(211)
        plt.plot(self.days_minute[:days, :].flatten())
        plt.subplot(212)
        plt.plot(self.days_quarter[:days, :].flatten())
        plt.show()

if __name__ == '__main__':
    weather = Weather(season='5winterdays')
    weather.plot(days=60)
    print weather.days_minute.shape