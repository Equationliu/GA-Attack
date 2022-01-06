import torch
import math
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import Encoder, Decoder

class StyleTransferNet(object):

    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def decode(self, x):
        img = self.decoder.decode(x)
        # post processing for output of decoder
        img = self.encoder.deprocess(img)
        return img

    def encode(self, img):
        # Note that the pretrained vgg model accepts BGR format, but the function by default take RGB value
        img = self.encoder.preprocess(img)
        x = self.encoder.encode(img)
        return x

def moments(content):
    meanC = content.mean(dim=(2, 3), keepdims=True)
    square_mean = torch.square(content).mean(dim=(2, 3), keepdims=True)
    varC = square_mean - meanC.square()
    return meanC, varC

stn = StyleTransferNet()



def DI_FSA(model, x_nature, y, args, loss_fn, random_init=False):
    batch_size = x_nature.shape[0]
    eps = args.epsilon
    num_steps = int(0.5 * args.num_steps * (1 + (args.intervals * math.log(args.epsilon)) / math.log(args.max_epsilon)))
    alpha = 1.25 * eps / args.num_steps

    # Natural statistics
    img_scaled = 255 * x_nature

    # encode image
    enc_c, _ = stn.encode(img_scaled)
    enc_c.requires_grad_(False)
    generated_img = stn.decode(enc_c)
    generated_img_rescaled = generated_img / 255.0

    with torch.no_grad():
        out = model(generated_img_rescaled, diversity=False)

    reconstruct_err = (out.data.max(1)[1] != y.data).float().sum()
    print("Batch error after Image Reconstruction: ", reconstruct_err.item())
    
    meanC, varC = moments(enc_c)
    sigmaC = torch.sqrt(varC + 1e-5)
    sign = torch.sign(meanC)
    abs_meanC = torch.abs(meanC) + 1e-6

    limit = 10 / torch.sqrt(torch.tensor(128.0).cuda())

    if random_init:
        meanC_delta_rand = torch.distributions.uniform.Uniform(low = abs_meanC / eps, high = eps * abs_meanC)
        sigmaC_delta_rand = torch.distributions.uniform.Uniform(sigmaC / eps, eps * sigmaC)
        meanS_delta = Variable(meanC_delta_rand.sample(), requires_grad=True)
        sigmaS_delta = Variable(sigmaC_delta_rand.sample(), requires_grad=True)
    else:
        meanS_delta = Variable(abs_meanC.clone(), requires_grad=True)
        sigmaS_delta = Variable(sigmaC.clone(), requires_grad=True)

    for idx in range(num_steps):

        target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta

        # decode target features back to image
        generated_adv_img = stn.decode(target_features)
        enc_gen_adv, _ = stn.encode(generated_adv_img)
        generated_adv_img_rescaled = generated_adv_img / 255.0

        with torch.enable_grad():
            model_input = F.interpolate(generated_adv_img_rescaled, size=299, mode="nearest")
            logits = model(model_input, diversity=True)
            adv_loss = loss_fn(logits, y)

            # content loss
            content_loss = torch.square(enc_gen_adv - target_features).mean(dim = (2,3)).sum(dim = 1)
            adv_loss_total = adv_loss * batch_size * 128
            loss = content_loss + adv_loss_total

            if (idx + 1) % 10 == 0:
                print("adv_loss_total: ", adv_loss_total)
                print("content_loss: ", content_loss)
                target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
                generated_adv_img = stn.decode(target_features.detach()) / 255.0

                with torch.no_grad():
                    out = model(generated_adv_img, diversity=False)

                batch_err = (out.data.max(1)[1] != y.data).float().sum()
                print("Batch error ", batch_err.item())
        
        [meanS_delta_grad, sigmaS_delta_grad] = torch.autograd.grad(loss.sum(), [meanS_delta, sigmaS_delta])

        # gradient clipping
        meanS_delta_grad = meanS_delta_grad.detach().clamp_(-1 / torch.sqrt(limit), 1 / torch.sqrt(limit))
        sigmaS_delta_grad = sigmaS_delta_grad.detach().clamp_(-1 / torch.sqrt(limit), 1 / torch.sqrt(limit))

        meanS_delta = Variable(meanS_delta.data - alpha * torch.sign(meanS_delta_grad), requires_grad=True)
        sigmaS_delta = Variable(sigmaS_delta.data - alpha * torch.sign(sigmaS_delta_grad), requires_grad=True)

        # clip
        meanS_delta = Variable(torch.max(torch.min(meanS_delta.data, eps * abs_meanC), abs_meanC / eps), requires_grad=True)
        sigmaS_delta = Variable(torch.max(torch.min(sigmaS_delta.data, eps * sigmaC), sigmaC / eps), requires_grad=True)

    target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
    generated_adv_img = stn.decode(target_features.detach()) / 255.0


    # budget
    enlarge_mean = meanS_delta / abs_meanC
    enlarge_sigma = sigmaS_delta / sigmaC
    upper_bound_mean = torch.where(enlarge_mean >= 1.0, enlarge_mean, 1 / enlarge_mean)
    upper_bound_sigma = torch.where(enlarge_sigma >= 1.0, enlarge_sigma, 1 / enlarge_sigma)

    budget = torch.max(torch.amax(upper_bound_mean, dim=(1,2,3)), torch.amax(upper_bound_sigma, dim=(1,2,3)))

    return F.interpolate(generated_adv_img, size=299, mode="nearest"), budget



def DMI_FSA(model, x_nature, y, args, loss_fn, random_init=False):
    batch_size = x_nature.shape[0]
    eps = args.epsilon
    num_steps = int(0.5 * args.num_steps * (1 + (args.intervals * math.log(args.epsilon)) / math.log(args.max_epsilon)))
    alpha = 1.25 * eps / args.num_steps

    # Natural statistics
    img_scaled = 255 * x_nature

    # encode image
    enc_c, _ = stn.encode(img_scaled)
    enc_c.requires_grad_(False)
    generated_img = stn.decode(enc_c)
    generated_img_rescaled = generated_img / 255.0

    with torch.no_grad():
        out = model(generated_img_rescaled, diversity=False)

    reconstruct_err = (out.data.max(1)[1] != y.data).float().sum()
    print("Batch error after Image Reconstruction: ", reconstruct_err.item())
    
    meanC, varC = moments(enc_c)
    sigmaC = torch.sqrt(varC + 1e-5)
    sign = torch.sign(meanC)
    abs_meanC = torch.abs(meanC) + 1e-6

    limit = 10 / torch.sqrt(torch.tensor(128.0).cuda())

    if random_init:
        meanC_delta_rand = torch.distributions.uniform.Uniform(low = abs_meanC / eps, high = eps * abs_meanC)
        sigmaC_delta_rand = torch.distributions.uniform.Uniform(sigmaC / eps, eps * sigmaC)
        meanS_delta = Variable(meanC_delta_rand.sample(), requires_grad=True)
        sigmaS_delta = Variable(sigmaC_delta_rand.sample(), requires_grad=True)
    else:
        meanS_delta = Variable(abs_meanC.clone(), requires_grad=True)
        sigmaS_delta = Variable(sigmaC.clone(), requires_grad=True)

    # momentum 
    g_meanS_delta = torch.zeros_like(meanS_delta).cuda()
    g_sigmaS_delta = torch.zeros_like(sigmaS_delta).cuda()

    for idx in range(num_steps):

        target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta

        # decode target features back to image
        generated_adv_img = stn.decode(target_features)
        enc_gen_adv, _ = stn.encode(generated_adv_img)
        generated_adv_img_rescaled = generated_adv_img / 255.0

        with torch.enable_grad():
            model_input = F.interpolate(generated_adv_img_rescaled, size=299, mode="nearest")

            logits = model(model_input, diversity=True)
            adv_loss = loss_fn(logits, y)

            # content loss
            content_loss = torch.square(enc_gen_adv - target_features).mean(dim = (2,3)).sum(dim = 1)
            adv_loss_total = adv_loss * batch_size * 128
            loss = content_loss + adv_loss_total

            if (idx + 1) % 10 == 0:
                print("adv_loss_total: ", adv_loss_total)
                print("content_loss: ", content_loss)
                target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
                generated_adv_img = stn.decode(target_features.detach()) / 255.0

                with torch.no_grad():
                    out = model(generated_adv_img, diversity=False)

                batch_err = (out.data.max(1)[1] != y.data).float().sum()
                print("Batch error ", batch_err.item())
        
        [meanS_delta_grad, sigmaS_delta_grad] = torch.autograd.grad(loss.sum(), [meanS_delta, sigmaS_delta])

        # gradient clipping
        meanS_delta_grad = meanS_delta_grad.detach().clamp_(-1 / torch.sqrt(limit), 1 / torch.sqrt(limit))
        sigmaS_delta_grad = sigmaS_delta_grad.detach().clamp_(-1 / torch.sqrt(limit), 1 / torch.sqrt(limit))

        noise_mean = meanS_delta_grad / torch.abs(meanS_delta_grad).mean(dim=(1, 2, 3), keepdim=True)
        g_meanS_delta = g_meanS_delta * args.momentum + noise_mean
        meanS_delta = Variable(meanS_delta.data - alpha * torch.sign(g_meanS_delta), requires_grad=True)

        noise_sigma = sigmaS_delta_grad / torch.abs(sigmaS_delta_grad).mean(dim=(1, 2, 3), keepdim=True)
        g_sigmaS_delta = g_sigmaS_delta * args.momentum + noise_sigma
        sigmaS_delta = Variable(sigmaS_delta.data - alpha * torch.sign(g_sigmaS_delta), requires_grad=True)

        # clip
        meanS_delta = Variable(torch.max(torch.min(meanS_delta.data, eps * abs_meanC), abs_meanC / eps), requires_grad=True)
        sigmaS_delta = Variable(torch.max(torch.min(sigmaS_delta.data, eps * sigmaC), sigmaC / eps), requires_grad=True)

    target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
    generated_adv_img = stn.decode(target_features.detach()) / 255.0

    # budget
    enlarge_mean = meanS_delta / abs_meanC
    enlarge_sigma = sigmaS_delta / sigmaC
    upper_bound_mean = torch.where(enlarge_mean >= 1.0, enlarge_mean, 1 / enlarge_mean)
    upper_bound_sigma = torch.where(enlarge_sigma >= 1.0, enlarge_sigma, 1 / enlarge_sigma)

    budget = torch.max(torch.amax(upper_bound_mean, dim=(1,2,3)), torch.amax(upper_bound_sigma, dim=(1,2,3)))

    return F.interpolate(generated_adv_img, size=299, mode="nearest"), budget




def GA_DMI_FSA(model, eval_model, x_nature, y, args, loss_fn):
    batch_size = x_nature.shape[0]
    eps_list = [math.exp((idx + 1) * (math.log(args.max_epsilon) / args.intervals))for idx in range(args.intervals)]
    mask = torch.ones((batch_size, )).bool()
    num_steps = args.num_steps

    img_scaled = 255 * x_nature

    # encode image
    enc_c, _ = stn.encode(img_scaled)
    enc_c.requires_grad_(False)
    generated_img = stn.decode(enc_c)
    generated_img_rescaled = generated_img / 255.0

    with torch.no_grad():
        out = model(generated_img_rescaled, diversity=False)

    reconstruct_err = (out.data.max(1)[1] != y.data).float().sum()
    print("Batch error after Image Reconstruction: ", reconstruct_err.item())
    
    meanC, varC = moments(enc_c)
    sigmaC = torch.sqrt(varC + 1e-5)
    sign = torch.sign(meanC)
    abs_meanC = torch.abs(meanC) + 1e-6

    limit = 10 / torch.sqrt(torch.tensor(128.0).cuda())
    
    meanS_delta = Variable(abs_meanC.clone(), requires_grad=True)
    sigmaS_delta = Variable(sigmaC.clone(), requires_grad=True)

    # momentum 
    g_meanS_delta = torch.zeros_like(meanS_delta).cuda()
    g_sigmaS_delta = torch.zeros_like(sigmaS_delta).cuda()

    
    for eps in eps_list:
        step_size = 1.25 * eps / num_steps
        
        for idx in range(num_steps):

            target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta

            # decode target features back to image
            generated_adv_img = stn.decode(target_features)
            enc_gen_adv, _ = stn.encode(generated_adv_img)
            generated_adv_img_rescaled = generated_adv_img / 255.0

            with torch.enable_grad():
                model_input = F.interpolate(generated_adv_img_rescaled, size=299, mode="nearest")
                logits = model(model_input, diversity=True)
                adv_loss = loss_fn(logits, y)

                # content loss
                content_loss = torch.square(enc_gen_adv - target_features).mean(dim = (2,3)).sum(dim = 1)
                adv_loss_total = adv_loss * batch_size * 128
                loss = content_loss + adv_loss_total

                if (idx + 1) % 10 == 0:
                    print("adv_loss_total: ", adv_loss_total)
                    print("content_loss: ", content_loss)
                    target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
                    generated_adv_img = stn.decode(target_features.detach()) / 255.0

                    with torch.no_grad():
                        out = model(generated_adv_img, diversity=False)

                    batch_err = (out.data.max(1)[1] != y.data).float().sum()
                    print("Budget now: {}, Batch error: {}".format(eps, batch_err.item()))
            
            [meanS_delta_grad, sigmaS_delta_grad] = torch.autograd.grad(loss.sum(), [meanS_delta, sigmaS_delta])

            # gradient clipping
            meanS_delta_grad = meanS_delta_grad.detach().clamp_(-1 / torch.sqrt(limit), 1 / torch.sqrt(limit))
            sigmaS_delta_grad = sigmaS_delta_grad.detach().clamp_(-1 / torch.sqrt(limit), 1 / torch.sqrt(limit))

            noise_mean = meanS_delta_grad / torch.abs(meanS_delta_grad).mean(dim=(1, 2, 3), keepdim=True)
            g_meanS_delta[mask] = g_meanS_delta[mask] * args.momentum + noise_mean[mask]
            meanS_delta = Variable(meanS_delta.data - step_size * torch.sign(g_meanS_delta), requires_grad=True)

            noise_sigma = sigmaS_delta_grad / torch.abs(sigmaS_delta_grad).mean(dim=(1, 2, 3), keepdim=True)
            g_sigmaS_delta[mask] = g_sigmaS_delta[mask] * args.momentum + noise_sigma[mask]
            sigmaS_delta = Variable(sigmaS_delta.data - step_size * torch.sign(g_sigmaS_delta), requires_grad=True)

            # clip
            meanS_delta = Variable(torch.max(torch.min(meanS_delta.data, eps * abs_meanC), abs_meanC / eps), requires_grad=True)
            sigmaS_delta = Variable(torch.max(torch.min(sigmaS_delta.data, eps * sigmaC), sigmaC / eps), requires_grad=True)

        g_meanS_delta.zero_()
        g_sigmaS_delta.zero_()
        with torch.no_grad():
            target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
            generated_adv_img = stn.decode(target_features.detach()) / 255.0
            output = eval_model(generated_adv_img, diversity=False).detach()
        prob = F.softmax(output, dim=1)
        conf = prob[np.arange(batch_size), y.long()]
        mask = (conf >= args.thres)

        # early stopping
        if mask.sum() == 0:
            break

    target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
    generated_adv_img = stn.decode(target_features.detach()) / 255.0

    # budget
    enlarge_mean = meanS_delta / abs_meanC
    enlarge_sigma = sigmaS_delta / sigmaC
    upper_bound_mean = torch.where(enlarge_mean >= 1.0, enlarge_mean, 1 / enlarge_mean)
    upper_bound_sigma = torch.where(enlarge_sigma >= 1.0, enlarge_sigma, 1 / enlarge_sigma)

    budget = torch.max(torch.amax(upper_bound_mean, dim=(1,2,3)), torch.amax(upper_bound_sigma, dim=(1,2,3)))

    return F.interpolate(generated_adv_img, size=299, mode="nearest"), budget







def GA_DI_FSA(model, eval_model, x_nature, y, args, loss_fn):
    batch_size = x_nature.shape[0]
    eps_list = [math.exp((idx + 1) * (math.log(args.max_epsilon) / args.intervals))for idx in range(args.intervals)]
    mask = torch.ones((batch_size, )).bool()
    num_steps = args.num_steps

    img_scaled = 255 * x_nature

    # encode image
    enc_c, _ = stn.encode(img_scaled)
    enc_c.requires_grad_(False)
    generated_img = stn.decode(enc_c)
    generated_img_rescaled = generated_img / 255.0

    with torch.no_grad():
        out = model(generated_img_rescaled, diversity=False)

    reconstruct_err = (out.data.max(1)[1] != y.data).float().sum()
    print("Batch error after Image Reconstruction: ", reconstruct_err.item())
    
    meanC, varC = moments(enc_c)
    sigmaC = torch.sqrt(varC + 1e-5)
    sign = torch.sign(meanC)
    abs_meanC = torch.abs(meanC) + 1e-6

    limit = 10 / torch.sqrt(torch.tensor(128.0).cuda())
    
    meanS_delta = Variable(abs_meanC.clone(), requires_grad=True)
    sigmaS_delta = Variable(sigmaC.clone(), requires_grad=True)

    # momentum 
    g_meanS_delta = torch.zeros_like(meanS_delta).cuda()
    g_sigmaS_delta = torch.zeros_like(sigmaS_delta).cuda()

    for eps in eps_list:
        step_size = 1.25 * eps / num_steps
        
        for idx in range(num_steps):

            target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta

            # decode target features back to image
            generated_adv_img = stn.decode(target_features)
            enc_gen_adv, _ = stn.encode(generated_adv_img)
            generated_adv_img_rescaled = generated_adv_img / 255.0

            with torch.enable_grad():
                model_input = F.interpolate(generated_adv_img_rescaled, size=299, mode="nearest")
                logits = model(model_input, diversity=True)
                adv_loss = loss_fn(logits, y)

                # content loss
                content_loss = torch.square(enc_gen_adv - target_features).mean(dim = (2,3)).sum(dim = 1)
                adv_loss_total = adv_loss * batch_size * 128
                loss = content_loss + adv_loss_total

                if (idx + 1) % 10 == 0:
                    print("adv_loss_total: ", adv_loss_total)
                    print("content_loss: ", content_loss)
                    target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
                    generated_adv_img = stn.decode(target_features.detach()) / 255.0

                    with torch.no_grad():
                        out = model(generated_adv_img, diversity=False)

                    batch_err = (out.data.max(1)[1] != y.data).float().sum()
                    print("Budget now: {}, Batch error: {}".format(eps, batch_err.item()))
            
            [meanS_delta_grad, sigmaS_delta_grad] = torch.autograd.grad(loss.sum(), [meanS_delta, sigmaS_delta])

            # gradient clipping
            meanS_delta_grad = meanS_delta_grad.detach().clamp_(-1 / torch.sqrt(limit), 1 / torch.sqrt(limit))
            sigmaS_delta_grad = sigmaS_delta_grad.detach().clamp_(-1 / torch.sqrt(limit), 1 / torch.sqrt(limit))

            g_meanS_delta[mask] =  meanS_delta_grad[mask].clone()
            meanS_delta = Variable(meanS_delta.data - step_size * torch.sign(g_meanS_delta), requires_grad=True)

            g_sigmaS_delta[mask] = sigmaS_delta_grad[mask].clone()
            sigmaS_delta = Variable(sigmaS_delta.data - step_size * torch.sign(g_sigmaS_delta), requires_grad=True)
            
            # clip
            meanS_delta = Variable(torch.max(torch.min(meanS_delta.data, eps * abs_meanC), abs_meanC / eps), requires_grad=True)
            sigmaS_delta = Variable(torch.max(torch.min(sigmaS_delta.data, eps * sigmaC), sigmaC / eps), requires_grad=True)



            g_meanS_delta.zero_()
            g_sigmaS_delta.zero_()

        with torch.no_grad():
            target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
            generated_adv_img = stn.decode(target_features.detach()) / 255.0
            output = eval_model(generated_adv_img, diversity=False).detach()
        prob = F.softmax(output, dim=1)
        conf = prob[np.arange(batch_size), y.long()]
        mask = (conf >= args.thres)


        # early stopping
        if mask.sum() == 0:
            break

    target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
    generated_adv_img = stn.decode(target_features.detach()) / 255.0

    # budget
    enlarge_mean = meanS_delta / abs_meanC
    enlarge_sigma = sigmaS_delta / sigmaC
    upper_bound_mean = torch.where(enlarge_mean >= 1.0, enlarge_mean, 1 / enlarge_mean)
    upper_bound_sigma = torch.where(enlarge_sigma >= 1.0, enlarge_sigma, 1 / enlarge_sigma)

    budget = torch.max(torch.amax(upper_bound_mean, dim=(1,2,3)), torch.amax(upper_bound_sigma, dim=(1,2,3)))

    return F.interpolate(generated_adv_img, size=299, mode="nearest"), budget







def Feature_Adam_Attack(model, x_nature, y, args, loss_fn, diversity=True, random_init=False):
    batch_size = x_nature.shape[0]
    eps = args.epsilon

    # Natural statistics
    img_scaled = 255 * x_nature

    # encode image
    enc_c, _ = stn.encode(img_scaled)
    enc_c.requires_grad_(False)
    generated_img = stn.decode(enc_c)
    enc_gen_adv, _ = stn.encode(generated_img)
    adv_content_loss = torch.square(enc_gen_adv - enc_c).mean(dim = (2,3)).sum(dim = 1)
    generated_img_rescaled = generated_img / 255.0

    with torch.no_grad():
        out = model(generated_img_rescaled, diversity=False)

    adv_mask = (out.data.max(1)[1] == y.data).float()

    reconstruct_err = (out.data.max(1)[1] != y.data).float().sum()
    print("Batch error after Image Reconstruction: ", reconstruct_err.item())
    
    meanC, varC = moments(enc_c)
    sigmaC = torch.sqrt(varC + 1e-5)
    sign = torch.sign(meanC)
    abs_meanC = torch.abs(meanC) + 1e-6

    limit = 10 / torch.sqrt(torch.tensor(128.0).cuda())

    if random_init:
        meanC_delta_rand = torch.distributions.uniform.Uniform(low = abs_meanC / eps, high = eps * abs_meanC)
        sigmaC_delta_rand = torch.distributions.uniform.Uniform(sigmaC / eps, eps * sigmaC)
        meanS_delta = Variable(meanC_delta_rand.sample(), requires_grad=True)
        sigmaS_delta = Variable(sigmaC_delta_rand.sample(), requires_grad=True)
    else:
        meanS_delta = Variable(abs_meanC.clone(), requires_grad=True)
        sigmaS_delta = Variable(sigmaC.clone(), requires_grad=True)

   
    opt = optim.Adam([meanS_delta, sigmaS_delta], lr=5e-3, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    # gradient clipping
    for p in opt.param_groups[0]['params']:
        p.register_hook(lambda grad: torch.clamp(grad, -1 / torch.sqrt(limit), 1 / torch.sqrt(limit)))

    
    # best adv img
    rst_img = generated_img_rescaled.clone()

    for idx in range(args.num_steps):

        opt.zero_grad()

        target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta

        # decode target features back to image
        generated_adv_img = stn.decode(target_features)
        enc_gen_adv, _ = stn.encode(generated_adv_img)
        generated_adv_img_rescaled = generated_adv_img / 255.0

        with torch.enable_grad():
            model_input = F.interpolate(generated_adv_img_rescaled, size=299, mode="nearest")

            logits = model(model_input, diversity=diversity)
            adv_loss = loss_fn(logits, y)

            # content loss
            content_loss = torch.square(enc_gen_adv - target_features).mean(dim = (2,3)).sum(dim = 1)
            adv_loss_total = adv_loss * batch_size * 128
            loss = content_loss + adv_loss_total
            loss.sum().backward()

        opt.param_groups[0]['lr'] = 5e-3 / (1 + 1e-3 * idx) # 0.0075 
        opt.step()

        # clip
        meanS_delta.data.copy_(torch.max(torch.min(meanS_delta.data, eps * abs_meanC), abs_meanC / eps))
        sigmaS_delta.data.copy_(torch.max(torch.min(sigmaS_delta.data, eps * sigmaC), sigmaC / eps))

        # Monitor the progress
        with torch.no_grad():
            target_features = (enc_c - meanC) * sigmaS_delta / sigmaC  + sign * meanS_delta
            decoded_img = stn.decode(target_features.detach()) 
            generated_adv_img = decoded_img / 255.0

            out = model(generated_adv_img, diversity=False)
            _adv_mask = (out.data.max(1)[1] == y.data).float()
            prob_truth = F.softmax(out, dim = 1)[np.arange(batch_size), y]

            # content loss
            enc_gen_adv, _ = stn.encode(decoded_img)
            _content_loss = torch.square(enc_gen_adv - target_features).mean(dim = (2,3)).sum(dim = 1)

            for j in range(batch_size):
                if _adv_mask[j] < adv_mask[j] or (_adv_mask[j] == adv_mask[j] and _content_loss[j] < adv_content_loss[j]):
                    rst_img[j] = generated_adv_img[j]
                    adv_mask[j] = _adv_mask[j]
                    adv_content_loss[j] = _content_loss[j]

            if (idx + 1) % 10 == 0:
                print("adv_loss_total: ", adv_loss_total)
                print("content_loss: ", content_loss)
                batch_err = adv_mask.sum()
                print("Batch error ", batch_size - batch_err.item())
                print("label softmax: ", prob_truth)

    return F.interpolate(rst_img, size=299, mode="nearest")