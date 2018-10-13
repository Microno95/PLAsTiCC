import keras
import tqdm
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


class TqdmCallback(keras.callbacks.Callback):
    def set_iter_params(self, num_epochs, batch_size, num_datapoints, loss_func, val_ratio):
        self.num_epochs = num_epochs
        self.loss_func = loss_func
        self.loss_name = loss_func.__name__ if callable(loss_func) else loss_func
        self.loss_type = "{:.4%}" if "percentage" in self.loss_name else "{:.4e}"
        self.batch_size = batch_size
        self.num_datapoints = num_datapoints
        self.tqdm_epoch = None
        self.tqdm_batch = None
        self.validation_ratio = val_ratio
        
    def set_metrics_to_print(self, *metrics):
        self.metrics_to_print = metrics
        self.metrics_shorthand = ["".join([j[0].upper() for j in i.split("_")]) for i in self.metrics_to_print]
        self.metrics_to_print = [(i, j) for i,j in zip(self.metrics_shorthand, self.metrics_to_print)]
        
    def create_iterators(self):
        if self.tqdm_epoch: self.tqdm_epoch.close()
        if self.tqdm_batch: self.tqdm_batch.close()
        self.tqdm_epoch = tqdm.tqdm_notebook(total=self.num_epochs)
        total_steps = int(((self.num_datapoints*(1.0 - self.validation_ratio))//self.batch_size + (self.num_datapoints%self.batch_size > 0)) * self.num_epochs)
        self.tqdm_batch = tqdm.tqdm_notebook(total=total_steps)
    
    def on_epoch_end(self, batch, logs={}):
        self.tqdm_epoch.update()
        self.tqdm_epoch.set_description("Loss " + self.loss_type.format(logs['val_loss'] / (100.0 if "percentage" in self.loss_name else 1.0)))
        
    def on_batch_end(self, batch, logs={}):
        self.tqdm_batch.update()
        self.tqdm_batch.set_description(" | ".join(["{}: {}".format(i[0], logs[i[1]]) for i in self.metrics_to_print]))

class TestSetPlotCallback(keras.callbacks.Callback):    
    def set_test_set(self, testX, testY):
        self.testX, self.testY = testX, testY
        self.figure = None
        
    def set_plot_freq(self, n=100):
        self.plot_frequency = n
        
    def create_plot(self, fig_kwargs={}, plot_kwargs={}, test_col='g', pred_col='b'):
        if self.figure:
            plt.close(self.figure)
        self.figure = plt.figure(**fig_kwargs)
        self.ax = self.figure.add_subplot(111)
        self.iter_count = 0
        plot_kwargs.update({'color': test_col})
        self.test_line, = self.ax.plot(self.testY, **plot_kwargs)
        plot_kwargs.update({'color': pred_col})
        self.pred_line, = self.ax.plot(self.testY, **plot_kwargs)
        self.pred_plot_kwargs = plot_kwargs
        plt.show()
        
    def on_epoch_end(self, batch, logs={}):
        self.iter_count += 1
        if self.iter_count % self.plot_frequency == 0:
            self.pred_line.remove()
            self.pred_line, = self.ax.plot(self.model.predict(self.testX), **(self.pred_plot_kwargs))
            self.figure.canvas.draw()
            self.figure.show()

class ModelMetricsPlotCallback(keras.callbacks.Callback):    
    def set_metrics(self, *metric_names):
        self.metric_names = metric_names
        self.figure = None
        self.metric_vals = dict([(i, []) for i in metric_names])
        self.epochs = []
        
    def set_plot_freq(self, n=100):
        self.plot_frequency = n
        
    def create_plot(self, fig_kwargs={}, plot_kwargs={}):
        if self.figure:
            plt.close(self.figure)
        self.figure = plt.figure(**fig_kwargs)
        self.ax = self.figure.add_subplot(111)
        self.metrics_lines = dict([(i, self.ax.plot(self.epochs, self.metric_vals[i], label=i, **plot_kwargs)[0]) for i in self.metric_names])
        self.metrics_colors = dict([(i, self.metrics_lines[i].get_color()) for i in self.metric_names])
        self.plot_kwargs = plot_kwargs
        self.iter_count = 0
        self.ax.legend()
        self.ax.grid(True)
        plt.show()
        
    def on_epoch_end(self, batch, logs={}):
        self.iter_count += 1
        if self.iter_count % self.plot_frequency == 0:
            self.epochs.append(self.iter_count)
            for i in self.metric_names:
                self.metric_vals[i].append(logs.get(i))
                self.metrics_lines[i].remove()
            self.metrics_lines = dict([(i, self.ax.plot(self.epochs, self.metric_vals[i], label=i, color=self.metrics_colors[i], **self.plot_kwargs)[0]) for i in self.metric_names])
            self.figure.canvas.draw()
            self.figure.show()

def custom_loss(y_true, y_pred):
    return K.max(K.square(y_pred - y_true))
