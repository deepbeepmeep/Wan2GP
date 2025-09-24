import torch
from .utils import *
from functools import partial

# Many thanks to the LanPaint team for this implementation (https://github.com/scraed/LanPaint/)

def _pack_latents(latents):
    batch_size, num_channels_latents, _, height, width = latents.shape 

    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def _unpack_latents(latents, height, width, vae_scale_factor=8):
    batch_size, num_patches, channels = latents.shape

    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents

class LanPaint():
    def __init__(self, NSteps = 5, Friction = 15, Lambda = 8, Beta = 1, StepSize = 0.15, IS_FLUX = True, IS_FLOW = False):
        self.n_steps = NSteps
        self.chara_lamb = Lambda
        self.IS_FLUX = IS_FLUX
        self.IS_FLOW = IS_FLOW
        self.step_size = StepSize
        self.friction = Friction
        self.chara_beta = Beta
        self.img_dim_size = None
    def add_none_dims(self, array):
        # Create a tuple with ':' for the first dimension and 'None' repeated num_nones times
        index = (slice(None),) + (None,) * (self.img_dim_size-1)
        return array[index]
    def remove_none_dims(self, array):
        # Create a tuple with ':' for the first dimension and 'None' repeated num_nones times
        index = (slice(None),) + (0,) * (self.img_dim_size-1)
        return array[index]
    def __call__(self, denoise, cfg_predictions, true_cfg_scale, cfg_BIG, x, latent_image, noise, sigma, latent_mask, n_steps=None, height =720, width = 1280, vae_scale_factor = 8):
        latent_image = _unpack_latents(latent_image, height=height, width=width, vae_scale_factor=vae_scale_factor)
        noise = _unpack_latents(noise, height=height, width=width, vae_scale_factor=vae_scale_factor)
        x = _unpack_latents(x, height=height, width=width, vae_scale_factor=vae_scale_factor)
        latent_mask = _unpack_latents(latent_mask, height=height, width=width, vae_scale_factor=vae_scale_factor)
        self.height = height
        self.width = width
        self.vae_scale_factor = vae_scale_factor
        self.img_dim_size = len(x.shape)
        self.latent_image = latent_image
        self.noise = noise
        if n_steps is None:
            n_steps = self.n_steps
        out = self.LanPaint(denoise, cfg_predictions, true_cfg_scale, cfg_BIG, x, sigma, latent_mask, n_steps, self.IS_FLUX, self.IS_FLOW)
        out = _pack_latents(out)
        return out
    def LanPaint(self, denoise, cfg_predictions, true_cfg_scale, cfg_BIG,  x, sigma, latent_mask, n_steps, IS_FLUX, IS_FLOW):
        if IS_FLUX:
            cfg_BIG = 1.0

        def double_denoise(latents, t):
            latents = _pack_latents(latents)
            noise_pred, neg_noise_pred = denoise(latents, true_cfg_scale)
            if noise_pred == None: return None, None
            predict_std = cfg_predictions(noise_pred, neg_noise_pred, true_cfg_scale, t)
            predict_std = _unpack_latents(predict_std, self.height, self.width, self.vae_scale_factor)
            if true_cfg_scale ==  cfg_BIG:
                predict_big = predict_std
            else:
                predict_big = cfg_predictions(noise_pred, neg_noise_pred, cfg_BIG, t)
                predict_big = _unpack_latents(predict_big, self.height, self.width, self.vae_scale_factor)
            return predict_std, predict_big
        
        if len(sigma.shape) == 0:
            sigma = torch.tensor([sigma.item()])
        latent_mask = 1 - latent_mask
        if IS_FLUX or IS_FLOW:
            Flow_t = sigma
            abt = (1 - Flow_t)**2 / ((1 - Flow_t)**2 + Flow_t**2 )
            VE_Sigma = Flow_t / (1 - Flow_t)
            #print("t", torch.mean( sigma ).item(), "VE_Sigma", torch.mean( VE_Sigma ).item())
        else:
            VE_Sigma = sigma 
            abt = 1/( 1+VE_Sigma**2 )
            Flow_t = (1-abt)**0.5 / ( (1-abt)**0.5 + abt**0.5  )
        # VE_Sigma, abt, Flow_t = current_times
        current_times =  (VE_Sigma, abt, Flow_t)
        
        step_size = self.step_size * (1 - abt)
        step_size = self.add_none_dims(step_size)
        # self.inner_model.inner_model.scale_latent_inpaint returns variance exploding x_t values
        # This is the replace step
        # x = x * (1 - latent_mask) +  self.inner_model.inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image)* latent_mask

        noisy_image  = self.latent_image  * (1.0 - sigma) + self.noise * sigma 
        x = x * (1 - latent_mask) +  noisy_image * latent_mask

        if IS_FLUX or IS_FLOW:
            x_t = x * ( self.add_none_dims(abt)**0.5 + (1-self.add_none_dims(abt))**0.5 )
        else:
            x_t = x / ( 1+self.add_none_dims(VE_Sigma)**2 )**0.5 # switch to variance perserving x_t values

        ############ LanPaint Iterations Start ###############
        # after noise_scaling, noise = latent_image + noise * sigma, which is x_t in the variance exploding diffusion model notation for the known region.
        args = None
        for i in range(n_steps):
            score_func = partial( self.score_model, y = self.latent_image, mask = latent_mask, abt = self.add_none_dims(abt), sigma = self.add_none_dims(VE_Sigma), tflow = self.add_none_dims(Flow_t), denoise_func = double_denoise )
            if score_func is None: return None
            x_t, args = self.langevin_dynamics(x_t, score_func , latent_mask, step_size , current_times, sigma_x = self.add_none_dims(self.sigma_x(abt)), sigma_y = self.add_none_dims(self.sigma_y(abt)), args = args)  
        if IS_FLUX or IS_FLOW:
            x = x_t / ( self.add_none_dims(abt)**0.5 + (1-self.add_none_dims(abt))**0.5 )
        else:
            x = x_t * ( 1+self.add_none_dims(VE_Sigma)**2 )**0.5 # switch to variance perserving x_t values
        ############ LanPaint Iterations End ###############
        # out is x_0
        # out, _ = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        # out = out * (1-latent_mask) + self.latent_image * latent_mask
        # return out
        return x

    def score_model(self, x_t, y, mask, abt, sigma, tflow, denoise_func):
        
        lamb = self.chara_lamb
        if self.IS_FLUX or self.IS_FLOW:
            # compute t for flow model, with a small epsilon compensating for numerical error.
            x = x_t / ( abt**0.5 + (1-abt)**0.5 ) # switch to Gaussian flow matching
            x_0, x_0_BIG = denoise_func(x, self.remove_none_dims(tflow))
            if x_0 is None: return None
        else:
            x = x_t * ( 1+sigma**2 )**0.5 # switch to variance exploding
            x_0, x_0_BIG = denoise_func(x, self.remove_none_dims(sigma))
            if x_0 is None: return None

        score_x = -(x_t - x_0)
        score_y =  - (1 + lamb) * ( x_t - y )  + lamb * (x_t - x_0_BIG)  
        return score_x * (1 - mask) + score_y * mask
    def sigma_x(self, abt):
        # the time scale for the x_t update
        return abt**0
    def sigma_y(self, abt):
        beta = self.chara_beta * abt ** 0
        return beta

    def langevin_dynamics(self, x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):
        # prepare the step size and time parameters
        with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
            step_sizes = self.prepare_step_size(current_times, step_size, sigma_x, sigma_y)
            sigma, abt, dtx, dty, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y = step_sizes
        # print('mask',mask.device)
        if torch.mean(dtx) <= 0.:
            return x_t, args
        # -------------------------------------------------------------------------
        # Compute the Langevin dynamics update in variance perserving notation
        # -------------------------------------------------------------------------
        #x0 = self.x0_evalutation(x_t, score, sigma, args)
        #C = abt**0.5 * x0 / (1-abt)
        A = A_x * (1-mask) + A_y * mask
        D = D_x * (1-mask) + D_y * mask
        dt = dtx * (1-mask) + dty * mask
        Gamma = Gamma_x * (1-mask) + Gamma_y * mask


        def Coef_C(x_t):
            x0 = self.x0_evalutation(x_t, score, sigma, args)
            C = (abt**0.5 * x0  - x_t )/ (1-abt) + A * x_t
            return C
        def advance_time(x_t, v, dt, Gamma, A, C, D):
            dtype = x_t.dtype
            with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
                osc = StochasticHarmonicOscillator(Gamma, A, C, D )
                x_t, v = osc.dynamics(x_t, v, dt )
            x_t = x_t.to(dtype)
            v = v.to(dtype)
            return x_t, v
        if args is None:
            #v = torch.zeros_like(x_t)
            v = None
            C = Coef_C(x_t)
            #print(torch.squeeze(dtx), torch.squeeze(dty))
            x_t, v = advance_time(x_t, v, dt, Gamma, A, C, D)
        else:
            v, C = args

            x_t, v = advance_time(x_t, v, dt/2, Gamma, A, C, D)

            C_new = Coef_C(x_t)
            v = v + Gamma**0.5 * ( C_new - C) *dt

            x_t, v = advance_time(x_t, v, dt/2, Gamma, A, C, D)

            C = C_new
  
        return x_t, (v, C)

    def prepare_step_size(self, current_times, step_size, sigma_x, sigma_y):
        # -------------------------------------------------------------------------
        # Unpack current times parameters (sigma and abt)
        sigma, abt, flow_t = current_times
        sigma = self.add_none_dims(sigma)
        abt = self.add_none_dims(abt)
        # Compute time step (dtx, dty) for x and y branches.
        dtx = 2 * step_size * sigma_x
        dty = 2 * step_size * sigma_y
        
        # -------------------------------------------------------------------------
        # Define friction parameter Gamma_hat for each branch.
        # Using dtx**0 provides a tensor of the proper device/dtype.

        Gamma_hat_x = self.friction **2 * self.step_size * sigma_x / 0.1 * sigma**0
        Gamma_hat_y = self.friction **2 * self.step_size * sigma_y / 0.1 * sigma**0
        #print("Gamma_hat_x", torch.mean(Gamma_hat_x).item(), "Gamma_hat_y", torch.mean(Gamma_hat_y).item())
        # adjust dt to match denoise-addnoise steps sizes
        Gamma_hat_x /= 2.
        Gamma_hat_y /= 2.
        A_t_x = (1) / ( 1 - abt ) * dtx / 2
        A_t_y =  (1+self.chara_lamb) / ( 1 - abt ) * dty / 2


        A_x = A_t_x / (dtx/2)
        A_y = A_t_y / (dty/2)
        Gamma_x = Gamma_hat_x / (dtx/2)
        Gamma_y = Gamma_hat_y / (dty/2)

        #D_x = (2 * (1 + sigma**2) )**0.5
        #D_y = (2 * (1 + sigma**2) )**0.5
        D_x = (2 * abt**0 )**0.5
        D_y = (2 * abt**0 )**0.5
        return sigma, abt, dtx/2, dty/2, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y



    def x0_evalutation(self, x_t, score, sigma, args):
        x0 = x_t + score(x_t)
        return x0