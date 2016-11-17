from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import time_series_generator as ts


trace1 = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
trace2 = go.Scatter(x=[20, 30, 40], y=[50, 60, 70])
trace3 = go.Scatter(x=[300, 400, 500], y=[600, 700, 800])
trace4 = go.Scatter(x=[4000, 5000, 6000], y=[7000, 8000, 9000])
trace4 = go.Scatter(x=range(100),y=ts.create_list(1,100,ts.uptrend(ts.sinx))[0])

fig = tools.make_subplots(rows=4, cols=1, subplot_titles=('Plot 1', 'Plot 2',
                                                          'Plot 3', 'Plot 4'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 4, 1)

fig['layout'].update(height=600, width=600, title='Multiple Subplots' +
                                                  ' with Titles')

plot_url = py.plot(fig, filename='make-subplots-multiple-with-title')