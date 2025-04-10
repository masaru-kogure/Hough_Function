##################################################
## Discripition
## This program calculates Hough functions.
## It is translated from the MATLAB code presented in Wang et al. (2016). 
## Unlike the original, this implementation can compute D0 tide functions,
## which were not supported in Wang et al. (2016).
## This code is developted in Python 3.11.3.
##################################################
## Terms and Conditions of Use  
## This program is free to use for academic, non-commercial purposes. 
## Modification of the code is not recommended; any moddifications are made at your own risk. 
## If used in publications, you must cite both Wang et al. (2016) and Kogure et al. (2025) (see the following reference).
## We strongly encourage users to contact us for discussion before using the result of this software in publications, 
## to prevent misuse or misinterpretation of the output.
## The developers and their affiliated organizations are not responsible for any damages arising from use of the software.
## Reference
## Wang, H., Boyd, J. P., and Akmaev, R. A.: On computation of Hough functions, Geosci. Model Dev., 9, 1477–1488, https://doi.org/10.5194/gmd-9-1477-2016, 2016.
##################################################
## Author: Masaru Kogure
## Version: 3.0.0
## Email: masarukogure@yonsei.ac.kr
## Date: Last Update: 2025/04/10
##################################################

import numpy as np
import scipy.linalg as la
from scipy.special import lpmv
from scipy.interpolate import interp1d
import math
from scipy.stats import linregress

# Class for numerical differentiation methods
class derivation:
    @staticmethod
    def central_diff(vari, dx):
        import numpy as np
        # Uniform central difference
        dvari = np.convolve(vari, [-1, 0, 1], mode = 'same')/(2.*dx)
        dvari[0] = (vari[1] - vari[0])/(dx) # forward diff at start
        dvari[-1] = (vari[-2] - vari[-1])/(dx)  # backward diff at end
        return dvari

    @staticmethod
    def uneven_central_diff(vari, x):
        import numpy as np
        # Central difference for uneven grid
        dvari = np.convolve(vari, [-1, 0, 1], mode = 'same')/(np.roll(x, 1) - np.roll(x, -1))
        dvari[0] = (vari[1] - vari[0])/(x[1] - x[0])
        dvari[-1] = (vari[-1] - vari[-2])/(x[-1] - x[-2]) 
        return dvari
    
    @staticmethod
    def interpol_diff(x, y):
        import numpy as np
        # Interpolated 3-point finite difference for non-uniform grids
        x01 = np.roll(x, -1) - x
        x02 = np.roll(x, -1) - np.roll(x, 1)
        x12 = x - np.roll(x, 1)

        # Adjusting edge values manually
        x01[0] = x[0] - x[1]
        x01[-1] = x[-3] - x[-2]
        x02[0] = x[0] - x[2]
        x02[-1] = x[-3] - x[-1]        
        x12[0] = x[1] - x[2]
        x12[-1] = x[-2] - x[-1]
        
        # Finite difference approximation
        dy = np.roll(y, -1) * x12 / (x01 * x02) + y * (1./x12 - 1./x01) - np.roll(y, 1) * x01 / (x02 * x12)
        
        # First and last point: manual 3-point stencil
        dy[0] = y[0] * (x01[0] + x02[0]) / (x01[0] * x02[0]) - y[1] * x02[0]/(x01[0] * x12[0]) + y[2] * x01[0] / (x02[0] * x12[0])
        dy[-1] = -y[-1-2] * x12[-1] / (x01[-1] * x02[-1]) + y[-1-1] * x02[-1]/(x01[-1] * x12[-1]) - y[-1] * (x02[-1] + x12[-1]) / (x02[-1] * x12[-1]) 
        return dy
        #https://www.nv5geospatialsoftware.com/docs/DERIV.html
    
    @staticmethod
    def fft_diff(vari, dx, w=1, idel_filter= []):
        #window function
        # https://numpy.org/doc/stable/reference/routines.window.html
        import numpy as np
        #X = np.arange(int((len(vari) - 1)/2)) + 1
        #is_N_even = (np.mod(len(vari),2) == 0)
        #if is_N_even:
        #    wave = np.hstack([0, X, len(vari)/2, -(len(vari)/2 + 1) + X])/(len(vari) * dx)
        #else:
        #    wave = np.hstack([0, X, -(len(vari)/2) + X])/(len(vari) * dx) 
        # Spectral derivative using FFT (suitable for periodic signals)
        wave = np.fft.fftfreq(len(vari),d=dx) * 2 * np.pi
        # Apply ideal filter (cut off high-frequency modes if needed)
        if idel_filter:
            nd = wave >= idel_filter
            wave[nd] = 0
        dvari = np.real(np.fft.ifft(np.fft.fft(vari) * 1j * wave * w))
        return dvari
        #butterworth filter
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html#scipy.signal.buttord


class houghfunction:
    @staticmethod
    def vertical_wavenumber(phase, z, n):
        # Estimates vertical wavenumber by linear regression over phase
        vwn = phase * np.nan # slope
        vwne = phase * np.nan # error of slope
        
        # Phase unwrapping (prevent 2π jumps)
        for i in range(len(phase)-1):
            if (np.abs((phase[i+1] - phase[i])/(z[i+1] - z[i])) > 2):
                phase[i+1::] = phase[i+1::] - 2 * np.pi
            if ((phase[i+1] - phase[i])/(z[i+1] - z[i]) < 0) & (z[i] < 60) & (z[i] > 20):
                phase[i+1::] = phase[i+1::] - 2 * np.pi 
        
        # Sliding window regression to estimate slope
        for ih in range(np.int16(n/2), np.int16(len(z)-n/2)):
            resutl = linregress(z[np.int16(ih-n/2):np.int16(ih+n/2)], phase[np.int16(ih-n/2):np.int16(ih+n/2)])
            vwn[ih] = resutl.slope
            vwne[ih] = resutl.stderr

        return vwn, vwne
    @staticmethod 
    def normalize(hough):
        # Normalize Hough mode using energy norm, then max abs
        #houghN = hough/np.sum(hough*hough)
        houghN = hough/np.max(np.abs(hough)) 
        return houghN
    @staticmethod
    def hough_expand( tide, gslat, lat, hough):    
        # Expand tide field into Hough basis
        func = interp1d(lat, tide, axis =1)
        tidegs = func(gslat) # interpolate to Gaussian grid
        houghcompo = np.sum(tidegs * hough[np.newaxis,:],axis = 1)/np.sum(hough ** 2)
        
        return houghcompo 
        
    @staticmethod
    def pmn_polynomial_value(mm, n, m, x):
        # Generate normalized associated Legendre polynomials
        cx = np.zeros((mm, n + 1))
        if m <= n:
            cx[:, m] = 1.0
            factor = 1.0
            for j in range(1, m+1):
                cx[:, m] *= -factor * np.sqrt(1.0 - x**2)
                #cx[:, m] = -factor * np.sqrt(1.0 - x**2) * cx[:, m]
                factor += 2.0

        if m + 1 <= n:
            cx[:, m + 1] = (2 * m + 1) * x * cx[:, m]

        for j in range(m + 2, n + 1):
            cx[:, j] = ((2 * j - 1) * x * cx[:, j - 1] + (-j - m + 1) * cx[:, j - 2]) / (j - m)

        for j in range(m, n + 1):
            factor = np.sqrt(((2 * j + 1) * math.factorial(j - m)) / (2.0 * math.factorial(j + m)))
            cx[:, j] *= factor

        return cx
    @staticmethod
    def lgwt(N,a,b):
        # Gauss-Legendre quadrature nodes and weights
        N=N-1
        N1=N+1 
        N2=N+2
        xu = np.linspace(-1, 1, N1).reshape(-1, 1)
        indices = np.arange(0, N + 1)
        y = np.cos((2. * indices + 1.) * np.pi / (2. * N + 2.)) + (0.27 / N1) * np.squeeze(np.sin(np.pi * xu * N / N2))
        L=np.zeros([N1,N2])
        #Lp=np.zeros([N1,N2])
        y0=2.

        while np.max(np.abs(y-y0))>np.spacing(1):
            
            L[:,0]=1
            #Lp[:,0]=0
            
            L[:,1]=y
            #Lp[:,1]=1
            
            for k in range(1,N1):
                L[:,k+1]=( (2*(k + 1) - 1)*y*L[:,k]-(k)*L[:,k-1] )/(k+1)
        
            Lp=(N2)*( L[:,N1-1]-y*L[:,N2-1] )/(1-y**2)
            
            y0=y
            y=y0-L[:,N2-1]/Lp

        # Linear map from[-1,1] to [a,b]
        x=(a*(1.-y)+b*(1.+y))/2      

        # Compute the weights
        w=(b-a)/((1-y**2)*Lp**2)*(N2/N1)**2
        return x, w
    @staticmethod 
    def hough_function(a=6.37e6, g=9.81, omega=2 * np.pi / (24 * 3600), s=1, sigma=0.5, N=62, nlat=94):
        import derivation as derive
        
        N2 = np.int32(N / 2)
        sf = s / sigma
        L = np.zeros(N)
        M = np.zeros(N)
        
        # Build tridiagonal matrices
        for r in range(s, N + s):
            i = r - s
            L[i] = np.sqrt((r + s + 1) * (r + s + 2) * (r - s + 1) * (r - s + 2)) / (
                (2 * r + 3) * np.sqrt((2 * r + 1) * (2 * r + 5)) * (sf - (r + 1) * (r + 2))
            )
            if (s == 2) and (r == 2):
                M[i] = -(
                    (sigma ** 2 * (sf - r * (r + 1))) / ((r * (r + 1)) ** 2)
                ) + (r + 2) ** 2 * (r + s + 1) * (r - s + 1) / (
                    (r + 1) ** 2 * (2 * r + 3) * (2 * r + 1) * (sf - (r + 1) * (r + 2))
                )
            else:
                try:
                    M[i] = -(
                        (sigma ** 2 * (sf - r * (r + 1))) / ((r * (r + 1)) ** 2)
                    ) + (r + 2) ** 2 * (r + s + 1) * (r - s + 1) / (
                        (r + 1) ** 2 * (2 * r + 3) * (2 * r + 1) * (sf - (r + 1) * (r + 2))
                    ) + (r - 1) ** 2 * (r ** 2 - s ** 2) / (
                        r ** 2 * (4 * r ** 2 - 1) * (sf - r * (r - 1))
                    )
                except ZeroDivisionError:
                    M[i] = np.finfo(np.float64).max#np.nan_to_num(np.inf)
            # Check for infinity and replace with realmax equivalent
        if np.isinf(M[i]):
            M[i] = np.finfo(np.float64).max

        # Construct matrices for eigenvalue problem
        f1 = np.zeros((N2, N2))
        f2 = np.zeros((N2, N2))

        for i in range(N2):
            f1[i, i] = M[2 * i]
            f2[i, i] = M[2 * i + 1]
            if i + 1 < N2:
                f1[i, i + 1] = L[2 * i] 
                f1[i + 1, i] = L[2 * i] 
                f2[i, i + 1] = L[2 * i + 1]
                f2[i + 1, i] = L[2 * i + 1]
        # Solve eigenvalue problems
        d1, v1 = la.eig(f1)
        d2, v2 = la.eig(f2)

        lamb1 = np.real(d1)
        ii = np.argsort(-lamb1)
        lamb1 = lamb1[ii]
        v1 = v1[:, ii]
        #ht1 = 4.*(a**2)*(omega**2)/g*lamb1/1000
        lamb2 = np.real(d2)
        ii = np.argsort(-lamb2)
        lamb2 = lamb2[ii]
        v2 = v2[:, ii]

        # enforce positive max entry. Normalize sign
        for i in range(v1.shape[1]):
            if np.max(np.abs(v1[:, i])) != 0:
                if v1[np.argmax(np.abs(v1[:, i])), i] < 0:
                    v1[:, i] *= -1

        for i in range(v2.shape[1]):
            if np.max(np.abs(v2[:, i])) != 0:
                if v2[np.argmax(np.abs(v2[:, i])), i] < 0:
                    v2[:, i] *= -1
        #ht2 = 4.*(a**2)*(omega**2)/g*lamb2/1000

        # Get Legendre polynomials
        #x, w = np.polynomial.legendre.leggauss(nlat)
        #x = x[::-1] 
        #w = w[::-1]
        x, w = houghfunction.lgwt(nlat,-1,1)
        prs = houghfunction.pmn_polynomial_value(nlat, N + s, s, x)

        # Build Hough functions
        h1 = np.zeros((nlat, N2))
        h2 = np.zeros((nlat, N2))

        for i in range(N2):
            for j in range(N2):
                i1 = 2 * j + s
                i2 = 2 * j + s + 1
                for ii in range(nlat):
                    h1[ii, i] += v1[j, i] * prs[ii, i1]
                    h2[ii, i] += v2[j, i] * prs[ii, i2]

        lamb = np.zeros(N)
        hough = np.zeros((nlat, N))

        for i in range(N2):
            for j in range(nlat):
                i1 = 2 * i
                i2 = 2 * i + 1
                lamb[i1] = lamb1[i]
                lamb[i2] = lamb2[i]
                hough[j, i1] = h1[j, i]
                hough[j, i2] = h2[j, i]

        # Sort by equivalent depth
        ii = np.argsort(1.0 / lamb)
        lamb = lamb[ii]
        hough = hough[:, ii]

        h = 4.0 * a ** 2 * omega ** 2 / g * lamb / 1000
        
        """something strange"""
        # compute Hough functions for wind components
        b1 = (np.float128(sigma)**2 - np.float128(x)**2) * np.sqrt(1.0 - np.float128(x)**2)
        b2 = np.sqrt(1.0 - np.float128(x)**2) / (np.float128(sigma)**2 - np.float128(x)**2)
        dhdx = hough * np.nan
        hough_u = hough * np.nan
        hough_v = hough * np.nan
        derive = derivation()  # Create an instance of the derivation class
        for i_mode in range(0, N):
            #dhdx[:, i_mode] = derive.uneven_central_diff(hough[:,i_mode], x)
            dhdx[:, i_mode] = derive.interpol_diff(x, hough[:,i_mode])
        hough_u = ((s / b1[:,None]) * hough) - ((b2[:,None] * x[:,None] / sigma) * dhdx)
        hough_v = ((s / sigma) * (x[:,None] / b1[:,None])) * hough - (b2[:,None] * dhdx)

        # Return dominant 6 modes near maximum depth
        if (min(h) <= 0):
            ind = np.argmax(h)
            hough = hough[:, ind-3:ind+3]
            hough_u = hough_u[:, ind-3:ind+3]
            hough_v = hough_v[:, ind-3:ind+3]
            h = h[ind-3:ind+3]
            dhdx = dhdx[:, ind-3:ind+3]
            return hough, dhdx, hough_u, hough_v, h, x
        else:
            ind = np.argmax(h)
            hough = hough[:, 0:6]
            hough_u = hough_u[:, 0:6]
            hough_v = hough_v[:, 0:6]
            h = h[0:6]
            dhdx = dhdx[:, 0:6]
            return hough, dhdx, hough_u, hough_v, h, x
        
""""""

