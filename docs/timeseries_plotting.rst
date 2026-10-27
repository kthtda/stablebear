Plotting
========

:py:func:`~stablebear.plotting.plot_timeseries` draws a time series as the
step function it evaluates to, with matplotlib. Each value holds from its
observation until the next one, and the last until the end of the domain.
The time axis is labeled with the unit of the time series. With
``markers=True``, each observation is marked with a dot, which also shows
the last value, held only at the end of the domain, here 4 at 3.5 s.

.. image:: _static/timeseries_plot_steps_light.png
   :width: 80%
   :class: only-light
   :alt: A time series drawn as a step function

.. image:: _static/timeseries_plot_steps_dark.png
   :width: 80%
   :class: only-dark
   :alt: A time series drawn as a step function

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_plotting_fig.py
      :pyobject: plot_timeseries_steps
      :language: python

Windows in place
----------------

A window is drawn where it was cut from its time series, at its
:py:attr:`~stablebear.timeseries.TimeSeries.origin` plus its own times, so
that it lines up with the series on the same axes. Here the window of 3 s
starting at 3 s is drawn over the series it was cut from.

.. image:: _static/timeseries_plot_window_in_place_light.png
   :width: 80%
   :class: only-light
   :alt: A window drawn over the time series it was cut from

.. image:: _static/timeseries_plot_window_in_place_dark.png
   :width: 80%
   :class: only-dark
   :alt: A window drawn over the time series it was cut from

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_timeseries_plotting_fig.py
      :pyobject: plot_timeseries_window_in_place
      :language: python
