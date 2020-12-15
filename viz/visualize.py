import torch
from viz.latent_traversal import LatentTraverser
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import numpy as np
from scipy import stats
import torch.nn.functional as F


def traversal_values(size):
    cdf_traversal = np.linspace(0.05, 0.95, size)
    return stats.norm.ppf(cdf_traversal)


def traversal_values_min_max(min, max, size):
    return np.linspace(min, max, size)


class Visualizer:
    def __init__(self, model, img_size, latent_spec):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.
        Parameters
        ----------
        model : VAE_model.models.VAE instance
        """
        if 'parallel' in str(type(model)):
            self.model = model.module
        else:
            self.model = model
        self.latent_spec = latent_spec
        self.latent_traverser = LatentTraverser(self.latent_spec)
        self.save_images = True  # If false, each method returns a tensor
        self.img_size = img_size
        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.is_both_continue = 'cont_var' in latent_spec
        self.is_both_discrete = 'disc_var' in latent_spec

        self.latent_var_dim = 0
        self.latent_class_dim = 0
        self.num_disc_latents = 0
        self.latent_var_dim = 0
        self.latent_class_dim = 0

        if self.is_both_continue:
            self.latent_var_dim = latent_spec['cont_var']
            self.latent_class_dim = latent_spec['cont_class']
            self.latent_dim = latent_spec['cont_var'] + latent_spec['cont_class']
        elif self.is_both_discrete:
            self.latent_var_dim += sum([dim for dim in self.latent_spec['disc_var']])
            self.latent_class_dim += sum([dim for dim in self.latent_spec['disc_class']])
            self.disc_dims_var = latent_spec['disc_var']
            self.disc_dims_class = latent_spec['disc_class']
            self.latent_dim = self.latent_var_dim + self.latent_class_dim
        else:
            self.latent_var_dim = latent_spec['cont'] if self.is_continuous else None
            self.latent_class_dim += sum([dim for dim in self.latent_spec['disc']])
            self.latent_dim = self.latent_var_dim + self.latent_class_dim

    def build_compare_reconstruction(self, size, data, input_data, x_recon):
        # Upper half of plot will contain data, bottom half will contain
        # reconstructions
        num_images = int(size[0] * size[1] / 2)
        if data.shape[1] == 3:
            originals = input_data[:num_images].cpu()
        else:
            originals = input_data[:num_images].cpu()
        reconstructions = x_recon.view(-1, *self.img_size)[:num_images].cpu()
        # If there are fewer examples given than spaces available in grid,
        # augment with blank images
        num_examples = originals.size()[0]
        if num_images > num_examples:
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])

        # Concatenate images and reconstructions
        comparison = torch.cat([originals, reconstructions])

        return comparison

    def reconstructions(self, data, size=(8, 8), both_continue=False, filename='recon.png',
                        both_discrete=False, partial_reconstruciton=False, is_partial_rand_class=False,
                        return_loss=False, is_E1=False):
        """
        Generates reconstructions of data through the model.
        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)
        size : tuple of ints
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even, so that upper half contains true data and
            bottom half contains reconstructions
            :param is_partial_rand_class:
            :param both_continue:
            :param data:
            :param size:
            :param filename:
        """
        batch_size = data.size(0)
        nb_pixels = self.img_size[1] * self.img_size[2]

        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = data
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        x_recon, recons_random_variability, recons_random_class, latent_representation, latent_sample, \
        latent_sample_variability, latent_sample_class, latent_sample_random_continue, prediction, pred_noised, \
        prediction_partial_rand_class, prediction_random_variability, prediction_random_class, \
        prediction_zc_pert_zd, prediction_zc_zd_pert, z_var, \
        z_var_reconstructed = self.model(input_data,
                                         reconstruction_rand=True,
                                         both_continue=both_continue,
                                         both_discrete=both_discrete,
                                         is_partial_rand_class=is_partial_rand_class,
                                         is_E1=is_E1)

        self.model.train()

        recons_random_variability_loss = 0
        recons_random_class_loss = 0

        if partial_reconstruciton:
            recons_random_variability_loss = F.mse_loss(recons_random_variability, input_data)
            recons_random_class_loss = F.mse_loss(recons_random_class, input_data)
            if return_loss:
                pass
            else:
                comparison_random_var = self.build_compare_reconstruction(size, data, input_data,
                                                                          recons_random_variability)
                comparison_random_class = self.build_compare_reconstruction(size, data, input_data,
                                                                            recons_random_class)

        recon_loss = F.mse_loss(x_recon, input_data)

        if return_loss:
            pass
        else:
            comparison = self.build_compare_reconstruction(size, data, input_data, x_recon)

        if return_loss:
            return recon_loss.item(), recons_random_variability_loss.item(), recons_random_class_loss.item()
        else:
            if self.save_images:
                if partial_reconstruciton:
                    save_image(comparison.data, filename + '_originals_data', row=size[0])
                    save_image(comparison_random_var.data, filename + '_random_var__data', row=size[0])
                    save_image(comparison_random_class.data, filename + '_random_class_data', row=size[0])
                else:
                    save_image(comparison.data, filename, nrow=size[0])
            else:
                if partial_reconstruciton:
                    return make_grid(comparison.data, nrow=size[0]), \
                           make_grid(comparison_random_var.data, nrow=size[0]), \
                           make_grid(comparison_random_class.data, nrow=size[0]), \
                           x_recon, recon_loss.item(), recons_random_variability_loss.item(), recons_random_class_loss.item()
                else:
                    return make_grid(comparison.data, nrow=size[
                        0]), None, None, x_recon, recon_loss.item(), recons_random_variability_loss, recons_random_class_loss

    def zvar_randn_generation(self, data, size=(8, 8), both_continue=False,
                              both_discrete=False, partial_reconstruciton=False, is_partial_rand_class=False,
                              return_loss=False, is_E1=False):

        batch_size = data.size(0)
        nb_pixels = self.img_size[1] * self.img_size[2]

        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = data
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        x_recon, recons_random_variability, recons_random_class, latent_representation, latent_sample, \
        latent_sample_variability, latent_sample_class, latent_sample_random_continue, prediction, pred_noised, \
        prediction_partial_rand_class, prediction_random_variability, prediction_random_class, \
        prediction_zc_pert_zd, prediction_zc_zd_pert, z_var, \
        z_var_reconstructed = self.model(input_data,
                                         reconstruction_rand=True,
                                         both_continue=both_continue,
                                         both_discrete=both_discrete,
                                         is_partial_rand_class=is_partial_rand_class,
                                         is_E1=is_E1)

        self.model.train()

        return

    def samples(self, size=(8, 8), filename='samples.png', both_continue=False, both_discrete=False,
                real_distribution=False, mu=None, var=None):
        """
        Generates samples from learned distribution by sampling prior and
        decoding.
        size : tuple of ints
        """
        # Get prior samples from latent distribution
        cached_sample_prior = self.latent_traverser.sample_prior
        self.latent_traverser.sample_prior = True
        prior_samples = self.latent_traverser.traverse_grid(size=size, both_continue=both_continue,
                                                            both_discrete=both_discrete)
        self.latent_traverser.sample_prior = cached_sample_prior

        if real_distribution:
            mu_var = mu[0]
            mu_struct = mu[1]
            sigma_var = np.abs(var[0])
            sigma_struct = np.abs(var[1])
            sample = []
            for i in range(self.latent_var_dim):
                sample_var = np.random.normal(mu_var[i], sigma_var[i], (size[0] * size[1]))
                sample.append(sample_var)

            for i in range(self.latent_class_dim):
                sample_struct = np.random.normal(mu_struct[i], sigma_struct[i], (size[0] * size[1]))
                sample.append(sample_struct)

            sample = torch.tensor(sample)
            sample = sample.permute(1, 0)

        else:
            sample = np.random.normal(size=(size[0] * size[1], self.latent_dim))
        """
        logvar = torch.tensor(np.random.rand(size[0]*size[1], self.latent_dim))
        mu = torch.tensor(np.random.rand(size[0] * size[1], self.latent_dim))
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        sample = mu + std * eps
        """

        # sample = np.repeat(sample, size, axis=0).reshape((nb_samples, size, self.latent_dim))
        # Map samples through decoder
        prior_samples = torch.tensor(sample).float()
        generated = self._decode_latents(prior_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1]), generated
        else:
            return make_grid(generated.data, nrow=size[1]), generated

    def latent_traversal_line(self, cont_idx=None, disc_idx=None, size=8,
                              filename='traversal_line.png'):
        """
        Generates an image traversal through a latent dimension.
        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                             disc_idx=disc_idx,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size)

    def latent_traversal_grid(self, cont_idx=None, cont_axis=None,
                              disc_idx=None, disc_axis=None, size=(5, 5), batch_chairs=None, real_im=False,
                              nb_samples=1,
                              filename='traversal_grid.png', both_continue=False, both_discrete=False):
        """
        Generates a grid of image traversals through two latent dimensions.
        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
                                                             cont_axis=cont_axis,
                                                             disc_idx=disc_idx,
                                                             disc_axis=disc_axis,
                                                             size=size,
                                                             batch_chairs=batch_chairs,
                                                             real_im=real_im,
                                                             model=self.model,
                                                             nb_samples=nb_samples,
                                                             both_continue=both_continue,
                                                             both_discrete=both_discrete)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1])

    def latent_random_traversal(self, size=8, indx=0, nb_samples=5, both_continue=False, both_discrete=False):

        latent_samples = []
        samples = []

        sample_disc = []

        if both_continue:
            sample = np.random.normal(size=(nb_samples, self.latent_dim))
            sample = np.repeat(sample, size, axis=0).reshape((nb_samples, size, self.latent_dim))
        elif both_discrete:
            sample_disc_var = []
            sample_disc_class = []

            for i, disc_dim in enumerate(self.disc_dims_var):
                s = np.zeros((nb_samples, disc_dim))
                s[np.arange(nb_samples), np.random.randint(0, disc_dim, nb_samples)] = 1.
                sample_disc_var.append(torch.Tensor(s))

            for i, disc_dim in enumerate(self.disc_dims_class):
                s = np.zeros((nb_samples, disc_dim))
                s[np.arange(nb_samples), np.random.randint(0, disc_dim, nb_samples)] = 1.
                sample_disc_class.append(torch.Tensor(s))

            sample_disc_var = torch.cat(sample_disc_var, dim=1)
            sample_disc_class = torch.cat(sample_disc_class, dim=1)

            sample_disc_var = np.repeat(sample_disc_var, size, axis=0).reshape(
                (nb_samples, size, self.latent_var_dim))
            sample_disc_class = np.repeat(sample_disc_class, size, axis=0).reshape(
                (nb_samples, size, self.latent_class_dim))
            sample = np.concatenate((sample_disc_var, sample_disc_class), axis=2)
        else:
            if self.is_continuous:
                sample = np.random.normal(size=(nb_samples, self.latent_var_dim))

            if self.is_discrete:
                for i, disc_dim in enumerate(self.disc_dims):
                    s = np.zeros((nb_samples, disc_dim))
                    s[np.arange(nb_samples), np.random.randint(0, disc_dim, nb_samples)] = 1.
                    sample_disc.append(torch.Tensor(s))

                sample_disc = torch.cat(sample_disc, dim=1)

            sample = np.repeat(sample, size, axis=0).reshape((nb_samples, size, self.latent_var_dim))
            sample_disc = np.repeat(sample_disc, size, axis=0).reshape((nb_samples, size, self.latent_class_dim))
            sample = np.concatenate((sample, sample_disc), axis=2)

        cont_traversal = traversal_values(size)

        for i in range(nb_samples):
            for j in range(size):
                sample[i][j, indx] = cont_traversal[j]

        samples.append(torch.Tensor(sample.reshape((nb_samples * (size), self.latent_dim))))

        latent_samples.append(torch.cat(samples, dim=1))
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))

        return make_grid(generated.data, nrow=size)

    def latent_real_img_traversal(self, batch_chairs, size=8, z_component_traversal=None,
                                  both_continue=False, indx_image=None,
                                  both_discrete=False, is_partial_rand_class=False, is_E1=False):
        """
        From real images: traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.
        :param both_continue:
        :param batch_chairs: real images tensor.
        :param size: size of linear space for traversal latent dimension
        :param indx: latent traversal index
        :param nb_samples: number of samples
        :return:
        """

        samples = []
        latent_samples = []

        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = batch_chairs
        if torch.cuda.is_available():
            input_data = input_data.cuda()

        x_recon, recons_random_variability, recons_random_class, latent_dist, latent_sample, \
        latent_sample_variability, latent_sample_class, latent_sample_random_continue, prediction, pred_noised, \
        prediction_partial_rand_class, prediction_random_variability, prediction_random_class, \
        prediction_zc_pert_zd, prediction_zc_zd_pert, z_var, \
        z_var_reconstructed = self.model(input_data,
                                         reconstruction_rand=True,
                                         both_continue=both_continue,
                                         both_discrete=both_discrete,
                                         is_partial_rand_class=is_partial_rand_class,
                                         is_E1=is_E1)
        self.model.train()

        if indx_image is not None:
            indx = indx_image
        else:
            indx = np.random.randint(0, len(latent_sample))
        img_latent = latent_sample[indx]
        x_recons = x_recon[indx]
        nb_composante = len(z_component_traversal)
        img_latent = img_latent.unsqueeze(dim=0)

        sample = np.expand_dims(np.repeat(img_latent.detach().numpy(), size, axis=0), axis=0)  # shape: (1, size, z_dim)
        sample = np.repeat(sample, nb_composante, axis=0)  # shape: (nb_composante, size, z_dim)

        # cont_traversal = traversal_values(size)
        cont_traversal = traversal_values_min_max(np.min(sample) - np.abs(np.min(sample)),
                                                  np.max(sample) + np.abs(np.max(sample)),
                                                  size)  # shape: size

        indx_same_composante = np.argwhere(cont_traversal > 0)[0]

        for i, composante in enumerate(z_component_traversal):
            for j in range(size):
                sample[i][j, composante] = sample[i][j, composante] + cont_traversal[j]

        samples.append(torch.Tensor(sample.reshape((nb_composante * size, self.latent_dim))))

        latent_samples.append(torch.cat(samples, dim=1))
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))

        return make_grid(generated.data, nrow=size), x_recons, batch_chairs[indx], indx_same_composante

    def joint_latent_traversal(self, batch_chairs, size_struct=8, size_var=8,
                               both_continue=False, indx_image=None,
                               both_discrete=False, is_partial_rand_class=False, is_E1=False, real_img=False):

        if real_img:
            samples = []

            self.model.eval()
            # Pass data through VAE to obtain reconstruction
            with torch.no_grad():
                input_data = batch_chairs
            if torch.cuda.is_available():
                input_data = input_data.cuda()

            x_recon, recons_random_variability, recons_random_class, latent_dist, latent_sample, \
            latent_sample_variability, latent_sample_class, latent_sample_random_continue, prediction, pred_noised, \
            prediction_partial_rand_class, prediction_random_variability, prediction_random_class, \
            prediction_zc_pert_zd, prediction_zc_zd_pert, z_var, \
            z_var_reconstructed = self.model(input_data,
                                             reconstruction_rand=True,
                                             both_continue=both_continue,
                                             both_discrete=both_discrete,
                                             is_partial_rand_class=is_partial_rand_class,
                                             is_E1=is_E1)
            self.model.train()

            for i in range(size_struct):
                sample = []
                indx = np.random.randint(0, len(latent_sample))
                img_latent = latent_sample[indx]
                z_struct = img_latent[self.latent_class_dim:].unsqueeze(dim=0)
                z_struct_fixe = torch.tensor(np.expand_dims(np.repeat(z_struct.detach().numpy(), size_var, axis=0),
                                                            axis=0))  # shape: (1, size_var, latent_class_dim)
                z_var_rand = torch.randn((1, size_var, self.latent_var_dim))

                sample.append(z_var_rand)
                sample.append(z_struct_fixe)
                sample = torch.cat(sample, dim=2)  # shape: (1, size, latent_dim)
                samples.append(sample)

            samples = torch.cat(samples, dim=0)
            samples = torch.Tensor(samples.reshape((size_var * size_struct, self.latent_dim)))
            generated = self._decode_latents(samples.unsqueeze(dim=0))

        else:
            samples = []
            for i in range(size_struct):
                sample = []
                z_struct = torch.randn(size=(1, self.latent_class_dim))
                z_struct_fixe = torch.tensor(np.expand_dims(np.repeat(z_struct.detach().numpy(), size_var, axis=0),
                                        axis=0))  # shape: (1, size_var, latent_class_dim)
                z_var_rand = torch.randn((1, size_var, self.latent_var_dim))

                sample.append(z_var_rand)
                sample.append(z_struct_fixe)
                sample = torch.cat(sample, dim=2)  # shape: (1, size, latent_dim)
                samples.append(sample)

            samples = torch.cat(samples, dim=0)
            samples = torch.Tensor(samples.reshape((size_var*size_struct, self.latent_dim)))
            generated = self._decode_latents(samples.unsqueeze(dim=0))

        return make_grid(generated.data, nrow=size_var)

    """
    def joint_latent_real_img_traversal(self, batch_chairs, size=8, cont_indx=0, disc_indx=0, nb_samples=5):
        
        From real images: traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.
        :param batch_chairs: real images tensor.
        :param size: size of linear space for traversal latent dimension
        :param indx: latent traversal index
        :param nb_samples: number of samples
        :return:
        

        samples = []
        latent_samples = []

        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = batch_chairs
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        x_recon, latent_dist, latent_sample, prediction, prediction_random_continue = self.model(input_data)
        self.model.train()

        img_latent = latent_sample[:nb_samples]
        x_recons = x_recon[:nb_samples]

        sample = np.repeat(img_latent.detach().numpy(), size + 1, axis=0).reshape(
            (nb_samples, size + 1, self.latent_spec['cont']))
        cont_traversal = traversal_values(size)

        for i in range(nb_samples):
            for j in range(size):
                sample[i][j, cont_indx] = cont_traversal[j]

        if self.is_discrete:
            for i in range(nb_samples):
                for j in range(size):
                    samples[i][j, :] = 0
                    sample[i][j, disc_indx] = 1

        samples.append(torch.Tensor(sample.reshape((nb_samples * (size + 1), self.latent_spec['cont']))))

        latent_samples.append(torch.cat(samples, dim=1))
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))

        return make_grid(generated.data, nrow=size + 1), x_recons, batch_chairs[:nb_samples]
    """

    def all_latent_traversals(self, size=8, filename='all_traversals.png', both_continue=False, both_discrete=False):
        """
        Traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.
        Parameters
        ----------
        size : int
            Number of samples for each latent traversal.
            :param both_continue:
            :param size:
            :param filename:
        """
        latent_samples = []

        if self.is_both_continue:
            for cont_idx in range(self.latent_dim):
                latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                                          disc_idx=None,
                                                                          size=size,
                                                                          both_continue=both_continue,
                                                                          both_discrete=both_discrete))
        elif both_discrete:
            for disc_idx in range(self.model.num_disc_latents_var):
                latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                          disc_idx=disc_idx,
                                                                          size=size,
                                                                          both_discrete=both_discrete))
            for disc_idx in range(self.model.num_disc_latents_class):
                latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                          disc_idx=disc_idx,
                                                                          size=size,
                                                                          both_discrete=both_discrete))
        else:
            # Perform line traversal of every continuous and discrete latent
            for cont_idx in range(self.model.latent_cont_dim):
                latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                                          disc_idx=None,
                                                                          size=size))

            for disc_idx in range(self.model.num_disc_latents):
                latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                          disc_idx=disc_idx,
                                                                          size=size))

        generated = self._decode_latents(torch.cat(latent_samples, dim=0))

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size)

    def _decode_latents(self, latent_samples):
        """
        Decodes latent samples into images.
        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = Variable(latent_samples)
        if torch.cuda.is_available():
            latent_samples = latent_samples.cuda()
        return self.model._decode(latent_samples).cpu()
