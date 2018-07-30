import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from matplotlib import style
import scipy.fftpack as sp
from scipy import signal

style.use('ggplot')

class Index(object):
    
    def setFreq(self, event):
        x = Index().setIn()
        l.set_ydata(x)
        
        y = Index().setOut()
        l1.set_ydata(y)
        
        xf = sp.fft(x)
        lf.set_ydata(2.0/N * np.abs(xf[:N/2]))

        yf = sp.fft(y)
        l1f.set_ydata(2.0/N * np.abs(yf[:N/2]))
                
        plt.draw()
        

    def setIn(self):
        inp = np.sin(2*np.pi*sfreq.val*1000*t)
        return inp

    def setOut(self):
        x = Index().setIn()
        y = [0.0] * N
        for i in n:
            if i == 0:
                y[i] = 0.01209*x[i]
            if i == 1:
                y[i] = 0.01209*x[i] + 0.04836*x[i-1] + 2.145*y[i-1]
            if i == 2:
                y[i] = 0.01209*x[i] + 0.04836*x[i-1] + 0.07254*x[i-2] + 2.145*y[i-1] - 2.306*y[i-2]
            if i == 3:
                y[i] = 0.01209*x[i] + 0.04836*x[i-1] + 0.07254*x[i-2] + 0.04836*x[i-3] + 2.145*y[i-1] - 2.306*y[i-2] + 1.279*y[i-3]
            if i >= 4:
                y[i] = 0.01209*x[i] + 0.04836*x[i-1] + 0.07254*x[i-2] + 0.04836*x[i-3] + 0.01209*x[i-4] + 2.145*y[i-1] - 2.306*y[i-2] + 1.279*y[i-3] - 0.3121*y[i-4]
            
        return y

callback = Index()
axfreq = plt.axes([0.2, 0.025, 0.6, 0.05])
sfreq = Slider(axfreq, 'Frequency', 1, 200, 1, '%1.1f kHz')
sfreq.on_changed(callback.setFreq)

# Number of samplepoints
N = 1000
# sample period
T = 1.0 / 100000

t = np.linspace(0.0, N*T, N)
n = np.arange(0, N, 1)
f = np.linspace(0.0, 1.0/(2.0*T), N/2)

#Input signal
plt.subplot(3, 2, 1)
plt.subplots_adjust(bottom=0.3)
plt.subplots_adjust(hspace=0.5)
x = Index().setIn()
l, = plt.plot(t*1000, x, lw=2)
plt.title('Input signal')
plt.xlabel('Time (ms)')
plt.xlim([1, 2])


#Output signal
plt.subplot(3, 2, 3)
y = Index().setOut()
l1, = plt.plot(t*1000, y, lw=2)
plt.xlabel('Time (ms)')
plt.title('Output signal')
plt.xlim([1, 2])


#Input fft
plt.subplot(3, 2, 2)
xf = sp.fft(x)
lf, = plt.plot(f/1000, 2.0/N * np.abs(xf[:N/2]))
plt.title('Input frequency spectrum')
plt.xlabel('Frequency (kHz)')

#Output fft
plt.subplot(3, 2, 4)
yf = sp.fft(y)
l1f, = plt.plot(f/1000, 2.0/N * np.abs(yf[:N/2]))
plt.title('Output frequency spectrum')
plt.xlabel('Frequency (kHz)')


#Analog frequency Response
plt.subplot(3, 2, 5)
s1 = signal.lti([0.45618], [1/((50000/np.pi)*(50000/np.pi)*(50000/np.pi)*(50000/np.pi)),1.273/((50000/np.pi)*(50000/np.pi)*(50000/np.pi)), 1.8526/((50000/np.pi)*(50000/np.pi)), 1.1597/(50000/np.pi), 0.4395])
w, mag, phase = signal.bode(s1)
plt.title('Magnitude Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|G_(LP)| (dB)')

l2f, = plt.semilogx(w, mag)

plt.subplot(3, 2, 6)
l3f, = plt.semilogx(w, phase)
plt.title('Phase Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('phase(G_(LP)) (deg)')

#Digital frequency Response
w1 = np.linspace(1000.0, 1.0/(2.0*T), N/2)

norm = 1/50000

reBo = 0.0121 + 0.0482*np.cos(norm*np.pi*w1) + 0.0724*np.cos(2*norm*np.pi*w1) + 0.0482*np.cos(3*norm*np.pi*w1)+0.0121*np.cos(4*norm*np.pi*w1)
imBo = -(0.0482*np.sin(norm*np.pi*w1) + 0.0724*np.sin(2*norm*np.pi*w1) + 0.0482*np.sin(3*norm*np.pi*w1)+0.0121*np.sin(4*norm*np.pi*w1))

reOnd = 1 - 2.061*np.cos(norm*np.pi*w1) + 2.155*np.cos(2*norm*np.pi*w1) - 1.166*np.cos(3*norm*np.pi*w1) + 0.2727*np.cos(4*norm*np.pi*w1)
imOnd = 2.061*np.sin(norm*np.pi*w1) - 2.155*np.sin(2*norm*np.pi*w1) + 1.166*np.sin(3*norm*np.pi*w1) - 0.2727*np.sin(4*norm*np.pi*w1)

mag = (np.sqrt(np.power(reBo, 2) + np.power(imBo, 2)))/(np.sqrt(np.power(reOnd, 2) + np.power(imOnd, 2)))
phase = (np.arctan2(imBo,reBo) - np.arctan2(imOnd,reOnd))*180/np.pi

plt.subplot(3, 2, 5)
l4f, = plt.semilogx(w1, 20*np.log(mag))
plt.legend([l2f, l4f], ['Analog', 'Discrete'])

plt.subplot(3, 2, 6)
l5f, = plt.semilogx(w1, phase)


#print('#1 Backend:',plt.get_backend())
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()
