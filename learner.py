import torch
from plots import Loss_plot, Tsne_plot_images_separate, Image_generation, Confusion_matrix_gan, Softmax_plot_classifier, Softmax_plot_classifier_conditional
from plots import Attribution_plots, Tsne_plot_classifier, Confusion_matrix_classifier, TimePlot, Hist_plot, Image_generation_dataset, Attribution_plots_conditional
from plots import Fid_plot, Tsne_plot_images_combined, Tsne_plot_dis_cgan, Tsne_plot_dis_gan, Attribution_plots_classifier, Attribution_plots_classifier_conditional
from ccbdl.utils import DEVICE
from ccbdl.learning.gan import BaseGANLearning, BaseCGANLearning
from torcheval.metrics import FrechetInceptionDistance
from ccbdl.config_loader.loaders import ConfigurationLoader
from networks import CNN
import os


class Learner(BaseGANLearning):
    def __init__(self,
                 trial_path: str,
                 model,
                 train_data,
                 test_data,
                 val_data,
                 task,
                 learner_config: dict,
                 network_config: dict,
                 logging):

        super().__init__(train_data, test_data, val_data, trial_path, learner_config, task=task, logging=logging)

        self.model = model
        
        self.device = DEVICE
        print(self.device)
        
        self.figure_storage.dpi = 200
        
        def load_classifier_gray():
            config = ConfigurationLoader().read_config("config.yaml")
            if learner_config['layer'] == 'linear':
                classifier_config = config["classifier_linear"]
            
                classifier = CNN(1,"Classifier used for analysis", 
                              **classifier_config)
                checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_linear.pt")
            elif learner_config['layer'] == 'nlrl':
                classifier_config = config["classifier_nlrl"]
            
                classifier = CNN(1,"Classifier used for analysis", 
                              **classifier_config)
                checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_nlrl.pt")
            else:
                raise ValueError(
                    f"Invalid value for layer: {learner_config['layer']}, it should be 'linear' or 'nlrl'.")
            
            checkpoint = torch.load(checkpoint_path)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            return classifier
        
        self.classifier_gray = load_classifier_gray()
        
        def load_classifier():
            config = ConfigurationLoader().read_config("config.yaml")
            if learner_config['layer'] == 'linear':
                classifier_config = config["classifier_linear"]
            
                classifier = CNN(3,"Classifier to classify real and fake images", 
                              **classifier_config)
                checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_rgb_linear.pt")
            elif learner_config['layer'] == 'nlrl':
                classifier_config = config["classifier_nlrl"]
            
                classifier = CNN(3,"Classifier to classify real and fake images", 
                              **classifier_config)
                checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_rgb_nlrl.pt")
            else:
                raise ValueError(
                    f"Invalid value for layer: {learner_config['layer']}, it should be 'linear' or 'nlrl'.")
            
            checkpoint = torch.load(checkpoint_path)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            return classifier
        
        self.classifier = load_classifier()

        self.criterion_name = learner_config["criterion"]
        self.noise_dim = learner_config["noise_dim"]
        self.lr_exp = learner_config["learning_rate_exp"]
        self.lr_exp_l = learner_config["learning_rate_exp_l"]
        self.threshold = learner_config["threshold"]
        self.learner_config = learner_config
        self.network_config= network_config
        
        self.lr = 10**self.lr_exp
        self.lr_l = 10**self.lr_exp_l

        self.criterion = getattr(torch.nn, self.criterion_name)(reduction='mean').to(self.device)
        
        # Get the last layer's name
        last_layer_name_parts = list(self.model.discriminator.named_parameters())[-1][0].split('.')
        last_layer_name = last_layer_name_parts[0] + '.' + last_layer_name_parts[1]
        # print("Last layer name:", last_layer_name)
        
        # Separate out the parameters based on the last layer's name
        fc_params = [p for n, p in self.model.discriminator.named_parameters() if last_layer_name + '.' in n]  # Parameters of the last layer
        rest_params = [p for n, p in self.model.discriminator.named_parameters() if not last_layer_name + '.' in n]  # Parameters of layers before the last layer
        
        # print("FC Params:")
        # for p in fc_params:
        #     print(p.shape)
        # print("\nRest Params:")
        # for p in rest_params:
        #     print(p.shape)
        
        # print(self.model)

        self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr / 2, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(rest_params, lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_fc = torch.optim.Adam(fc_params, lr=self.lr_l)

        self.train_data = train_data
        self.test_data = test_data

        self.result_folder = trial_path

        self.plotter.register_default_plot(TimePlot(self))
        self.plotter.register_default_plot(Image_generation(self))
        self.plotter.register_default_plot(Image_generation_dataset(self))
        self.plotter.register_default_plot(Tsne_plot_classifier(self))
        self.plotter.register_default_plot(Attribution_plots(self))
        self.plotter.register_default_plot(Fid_plot(self))
        self.plotter.register_default_plot(Loss_plot(self))
        self.plotter.register_default_plot(Attribution_plots_classifier(self))
        
        if self.network_config["final_layer"] == 'nlrl':
            self.plotter.register_default_plot(Hist_plot(self))
        
        self.plotter.register_default_plot(Softmax_plot_classifier(self))
        self.plotter.register_default_plot(Confusion_matrix_gan(self))
        self.plotter.register_default_plot(Tsne_plot_dis_gan(self))

        self.parameter_storage.store(self)
        
        self.parameter_storage.write_tab(self.model.count_parameters(), "number of parameters:")
        self.parameter_storage.write_tab(self.model.count_learnable_parameters(), "number of learnable parameters: ")
        
        self.fid_metric = FrechetInceptionDistance(model=self.classifier, feature_dim=10, device=self.device)
        
        self.initial_save_path = os.path.join(self.result_folder, 'net_initial.pt')

    def _train_epoch(self, train=True):
        if self.epoch == 0:
            torch.save({'epoch': self.epoch,
                        'batch': 0,
                        'model_state_dict': self.model.state_dict()},
                       self.initial_save_path)
        self.model.train()
        for i, data in enumerate(self.train_data):
            inputs, labels = data
            
            self.real_data = inputs.to(self.device)
            self.labels = labels.to(self.device).long()
            
            # min_value = self.real_data.min()
            # max_value = self.real_data.max()
            
            # print(f"Min value: {min_value.item()}")
            # print(f"Max value: {max_value.item()}")
            
            self.optimizer_D.zero_grad()
            self.optimizer_D_fc.zero_grad()

            # Train discriminator on real data
            real_target = torch.ones(len(inputs), device=self.device)

            predictions_real = self._discriminate(self.real_data)

            diss_real = self.criterion(predictions_real, real_target)

            # Train discriminator on fake data
            noise = torch.randn(self.real_data.size(0), self.noise_dim, device=self.device)

            fake = self._generate(noise)
                
            fake_target = torch.zeros(len(inputs), device=self.device)
            
            predictions_fake = self._discriminate(fake.detach())

            dis_fake = self.criterion(predictions_fake, fake_target)

            self.loss_disc = diss_real + dis_fake
            
            self.loss_disc.backward()
            self.optimizer_D.step()
            self.optimizer_D_fc.step()

            self.optimizer_G.zero_grad()

            real_acc = 100.0 * (predictions_real > self.threshold).sum() / real_target.shape[0]
            fake_acc = 100.0 * (predictions_fake < self.threshold).sum() / fake_target.shape[0]
            self.train_accuracy = (real_acc + fake_acc)/2

            output = self._discriminate(fake)
            # min_value = output.min()
            # max_value = output.max()
            
            # print(f"Min value: {min_value.item()}")
            # print(f"Max value: {max_value.item()}")

            self.loss_gen = self.criterion(output, real_target)
            
            self.loss_gen.backward()
            self.optimizer_G.step()
            
            # Convert real and fake images from grayscale to RGB
            real_images_rgb = self.grayscale_to_rgb(self.real_data)
            fake_images_rgb = self.grayscale_to_rgb(fake.detach())

            # Update the metric for real images and fake images of RGB
            self.fid_metric.update(real_images_rgb, is_real=True)
            self.fid_metric.update(fake_images_rgb, is_real=False)
            
            self.fid = self.fid_metric.compute()

            self.data_storage.store(
                [self.epoch, self.batch, self.train_accuracy, self.test_accuracy, self.loss_disc, 
                 self.loss_gen, self.fid])

            if train:
                self.batch += 1
                self.data_storage.dump_store("predictions_real", predictions_real.detach().cpu().numpy())
                self.data_storage.dump_store("predictions_fake", predictions_fake.detach().cpu().numpy())
                self.data_storage.dump_store("fake_images", fake.detach())
                self.data_storage.dump_store("real_images", self.real_data)
                self.data_storage.dump_store("labels", self.labels)

    def _test_epoch(self, test=True):
        self.model.eval()
        loss, samples, corrects = 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                inputs, labels = data
                
                real_data = inputs.to(self.device)
                labels = labels.to(self.device).long()
                real_target = torch.ones(len(inputs), device=self.device)

                # Classify Images
                predictions = self._discriminate(real_data)
                # min_value = predictions.min()
                # max_value = predictions.max()
                
                # print(f"Min value: {min_value.item()}")
                # print(f"Max value: {max_value.item()}")

                loss += self.criterion(predictions, real_target).item()
                corrects += (predictions > self.threshold).sum()
                samples += inputs.size(0)
                
        self.test_loss = loss / len(self.test_data)
        self.test_accuracy = 100.0 * corrects / samples

    def _validate_epoch(self):
        pass

    def _generate(self, ins):
        return self.model.generator(ins)

    def _discriminate(self, ins):
        return self.model.discriminator(ins).squeeze()

    def _update_best(self):
        if self.fid < self.best_values["FidScore"]:
            self.best_values["GenLoss"] = self.loss_gen.item()
            self.best_values["DisLoss"] = self.loss_disc.item()
            self.best_values["FidScore"] = self.fid.item()
            self.best_values["Batch"] = self.batch

            self.other_best_values = {'testloss':      self.test_loss,
                                      "test_acc":      self.test_accuracy.item(),
                                      "train_acc":     self.train_accuracy.item(), }

            self.best_state_dict = self.model.state_dict()

    def evaluate(self):
        self.end_values = {"GenLoss":       self.loss_gen.item(),
                           "DisLoss":       self.loss_disc.item(),
                           "FidScore":      self.fid.item(),
                           'testloss':      self.test_loss,
                           "test_acc":      self.test_accuracy.item(),
                           "train_acc":     self.train_accuracy.item(),
                           "Batch":         self.batch}

    def _hook_every_epoch(self):
        if self.epoch == 0:
            self.init_values = {"GenLoss":       self.loss_gen.item(),
                                "DisLoss":       self.loss_disc.item(),
                                "FidScore":      self.fid.item(),
                                'testloss':      self.test_loss,
                                "test_acc":      self.test_accuracy.item(),
                                "train_acc":     self.train_accuracy.item(),
                                "Batch":         self.batch}

            self.init_state_dict = {'epoch': self.epoch,
                                    'batch': self.batch,
                                    'GenLoss': self.loss_gen.item(),
                                    'DisLoss': self.loss_disc.item(),
                                    'FidScore': self.fid.item(),
                                    'model_state_dict': self.model.state_dict()}
            self.test_noise = torch.randn(30, self.noise_dim, device=self.device)

        # Saving generated images
        with torch.no_grad(): 
            pos_epoch = self.epoch

            generated_images = self._generate(self.test_noise)
            predictions_gen = self._discriminate(generated_images.detach())

            self.data_storage.dump_store("generated_images", generated_images)
            self.data_storage.dump_store("epochs_gen", pos_epoch)
            self.data_storage.dump_store("predictions_gen", predictions_gen.detach().cpu().numpy())

    def _save(self):
        torch.save(self.init_state_dict, self.init_save_path)
        
        torch.save({'epoch': self.epoch,
                    'best_values': self.best_values,
                    'best_acc_loss_values': self.other_best_values,
                    'model_state_dict': self.best_state_dict},
                   self.best_save_path)

        torch.save({'epoch': self.epoch,
                    'batch': self.batch,
                    'GenLoss': self.loss_gen.item(),
                    'DisLoss': self.loss_disc.item(),
                    'FidScore': self.fid.item(),
                    'model_state_dict': self.model.state_dict()},
                   self.net_save_path)

        self.parameter_storage.store(self.init_values, "initial_values")
        self.parameter_storage.store(self.best_values, "best_values")
        self.parameter_storage.store(self.other_best_values, "best_acc_loss_values")
        self.parameter_storage.store(self.end_values, "end_values")
        self.parameter_storage.write("\n")
        # if self.best_values["FidScore"] <= 7.0:
        torch.save(self.data_storage, os.path.join(self.result_folder, "data_storage.pt"))
    
    def _load_initial(self):
        checkpoint = torch.load(self.initial_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model.discriminator
    
    def _load_best(self):
        checkpoint = torch.load(self.best_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model.discriminator
    
    def grayscale_to_rgb(self, images):
                # `images` is expected to be of shape [batch_size, 1, height, width]
                return images.repeat(1, 3, 1, 1)


class Conditional_Learner(BaseCGANLearning):
    def __init__(self,
                  trial_path: str,
                  model,
                  train_data,
                  test_data,
                  val_data,
                  task,
                  learner_config: dict,
                  network_config: dict,
                  logging):

        super().__init__(train_data, test_data, val_data, trial_path, learner_config, task=task, logging=logging)

        self.model = model
        
        self.device = DEVICE
        print(self.device)
        
        self.figure_storage.dpi = 200
        
        def load_classifier():
            config = ConfigurationLoader().read_config("config.yaml")
            if learner_config['layer'] == 'linear':
                classifier_config = config["classifier_linear"]
            
                classifier = CNN(3,"Classifier to classify real and fake images", 
                              **classifier_config)
                checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_rgb_linear.pt")
            elif learner_config['layer'] == 'nlrl':
                classifier_config = config["classifier_nlrl"]
            
                classifier = CNN(3,"Classifier to classify real and fake images", 
                              **classifier_config)
                checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_rgb_nlrl.pt")
            else:
                raise ValueError(
                    f"Invalid value for layer: {learner_config['layer']}, it should be 'linear' or 'nlrl'.")
            
            checkpoint = torch.load(checkpoint_path)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            return classifier
        
        self.classifier = load_classifier()

        self.criterion_name = learner_config["criterion"]
        self.noise_dim = learner_config["noise_dim"]
        self.threshold = learner_config["threshold"]
        self.lr_exp = learner_config["learning_rate_exp"]
        self.lr_exp_l = learner_config["learning_rate_exp_l"]
        self.learner_config = learner_config
        self.network_config = network_config
        
        self.lr = 10**self.lr_exp
        self.lr_l = 10**self.lr_exp_l

        self.criterion = getattr(torch.nn, self.criterion_name)(reduction='mean').to(self.device)

        # Get the last layer's name
        last_layer_name_parts = list(self.model.discriminator.named_parameters())[-1][0].split('.')
        last_layer_name = last_layer_name_parts[0] + '.' + last_layer_name_parts[1]
        print("Last layer name:", last_layer_name)
        
        # Separate out the parameters based on the last layer's name
        fc_params = [p for n, p in self.model.discriminator.named_parameters() if last_layer_name + '.' in n]  # Parameters of the last layer
        rest_params = [p for n, p in self.model.discriminator.named_parameters() if not last_layer_name + '.' in n]  # Parameters of layers before the last layer
        
        print("FC Params:")
        for p in fc_params:
            print(p.shape)
        print("\nRest Params:")
        for p in rest_params:
            print(p.shape)
        
        print(self.model)

        self.optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=self.lr / 2, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(rest_params, lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_fc = torch.optim.Adam(fc_params, lr=self.lr_l, betas=(0.5, 0.999))

        self.train_data = train_data
        self.test_data = test_data

        self.result_folder = trial_path

        self.plotter.register_default_plot(TimePlot(self))               
        # self.plotter.register_default_plot(Image_generation(self))
        # self.plotter.register_default_plot(Image_generation_dataset(self))       
        # self.plotter.register_default_plot(Attribution_plots_conditional(self))
        self.plotter.register_default_plot(Fid_plot(self))
        self.plotter.register_default_plot(Confusion_matrix_gan(self))
        # self.plotter.register_default_plot(Attribution_plots_classifier_conditional(self))
        
        # if self.network_config["final_layer"] == 'nlrl':
        #     self.plotter.register_default_plot(Hist_plot(self))
            
        # self.plotter.register_default_plot(Tsne_plot_images_separate(self))
        # self.plotter.register_default_plot(Tsne_plot_images_combined(self))  
        # self.plotter.register_default_plot(Tsne_plot_classifier(self))
        # self.plotter.register_default_plot(Tsne_plot_dis_cgan(self))      
        # self.plotter.register_default_plot(Softmax_plot_classifier_conditional(self))
        self.plotter.register_default_plot(Loss_plot(self))
        # self.plotter.register_default_plot(Confusion_matrix_classifier(self, **{"ticks": torch.arange(0, 10, 1).numpy()}))

        self.parameter_storage.store(self)
        
        self.parameter_storage.write_tab(self.model.count_parameters(), "number of parameters:")
        self.parameter_storage.write_tab(self.model.count_learnable_parameters(), "number of learnable parameters: ")
        
        self.fid_metric = FrechetInceptionDistance(model=self.classifier, feature_dim=10, device=self.device)
        
        self.initial_save_path = os.path.join(self.result_folder, 'net_initial.pt')

    def _train_epoch(self, train=True):
        if self.epoch == 0:
            torch.save({'epoch': self.epoch,
                        'batch': 0,
                        'model_state_dict': self.model.state_dict()},
                       self.initial_save_path)
        self.model.train()
        for i, data in enumerate(self.train_data):
            inputs, labels = data
            
            self.real_data = inputs.to(self.device)
            self.labels = labels.to(self.device).long()
            
            self.optimizer_D.zero_grad()
            self.optimizer_D_fc.zero_grad()

            # Train discriminator on real data
            real_target = torch.ones(len(inputs), device=self.device)

            predictions_real = self._discriminate(self.real_data, self.labels)

            diss_real = self.criterion(predictions_real, real_target)

            # Train discriminator on fake data
            noise = torch.randn(self.real_data.size(0), self.noise_dim, device=self.device)

            fake = self._generate(noise, self.labels)
                
            fake_target = torch.zeros(len(inputs), device=self.device)

            predictions_fake = self._discriminate(fake.detach(), self.labels)

            dis_fake = self.criterion(predictions_fake, fake_target)

            self.loss_disc = diss_real + dis_fake
            
            self.loss_disc.backward()
            self.optimizer_D.step()
            self.optimizer_D_fc.step()

            self.optimizer_G.zero_grad()

            real_acc = 100.0 * (predictions_real > self.threshold).sum() / real_target.shape[0]
            fake_acc = 100.0 * (predictions_fake < self.threshold).sum() / fake_target.shape[0]
            self.train_accuracy = (real_acc + fake_acc)/2

            # Fooling the discriminator with fake
            output = self._discriminate(fake, self.labels)

            self.loss_gen = self.criterion(output, real_target)

            self.loss_gen.backward()
            self.optimizer_G.step()
            
            # Convert real and fake images from grayscale to RGB
            real_images_rgb = self.grayscale_to_rgb(self.real_data)
            fake_images_rgb = self.grayscale_to_rgb(fake.detach())

            # Update the metric for real images
            self.fid_metric.update(real_images_rgb, is_real=True)
            # Update the metric for generated images
            self.fid_metric.update(fake_images_rgb, is_real=False)
            
            self.fid = self.fid_metric.compute()

            self.data_storage.store(
                [self.epoch, self.batch, self.train_accuracy, self.test_accuracy, self.loss_disc, 
                  self.loss_gen, self.fid])

            if train:
                self.batch += 1
                self.data_storage.dump_store("predictions_real", predictions_real.detach().cpu().numpy())
                self.data_storage.dump_store("predictions_fake", predictions_fake.detach().cpu().numpy())
                self.data_storage.dump_store("fake_images", fake.detach())
                self.data_storage.dump_store("real_images", self.real_data)
                self.data_storage.dump_store("labels", self.labels)

    def _test_epoch(self, test=True):        
        self.model.eval()
        loss, samples, corrects = 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                inputs, labels = data
                
                real_data = inputs.to(self.device)
                labels = labels.to(self.device).long()
                real_target = torch.ones(len(inputs), device=self.device)

                # Classify Images
                predictions = self._discriminate(real_data, labels)

                loss += self.criterion(predictions, real_target).item()
                corrects += (predictions > self.threshold).sum()
                samples += inputs.size(0)
                
        self.test_loss = loss / len(self.test_data)
        self.test_accuracy = 100.0 * corrects / samples

    def _validate_epoch(self):
        pass

    def _generate(self, ins, labels):
        return self.model.generator(ins, labels)

    def _discriminate(self, ins, labels):
        return self.model.discriminator(ins, labels).squeeze()

    def _update_best(self):
        if self.loss_gen < self.best_values["GenLoss"]:
            self.best_values["GenLoss"] = self.loss_gen.item()
            self.best_values["DisLoss"] = self.loss_disc.item()
            self.best_values["FidScore"] = self.fid.item()
            self.best_values["Batch"] = self.batch

            self.other_best_values = {'testloss':      self.test_loss,
                                      "test_acc":      self.test_accuracy.item(),
                                      "train_acc":     self.train_accuracy.item(), }

            self.best_state_dict = self.model.state_dict()

    def evaluate(self):
        self.end_values = {"GenLoss":       self.loss_gen.item(),
                            "DisLoss":       self.loss_disc.item(),
                            "FidScore":      self.fid.item(),
                            'testloss':      self.test_loss,
                            "test_acc":      self.test_accuracy.item(),
                            "train_acc":     self.train_accuracy.item(),
                            "Batch":         self.batch}

    def _hook_every_epoch(self):
        if self.epoch == 0:
            self.init_values = {"GenLoss":       self.loss_gen.item(),
                                "DisLoss":       self.loss_disc.item(),
                                "FidScore":      self.fid.item(),
                                'testloss':      self.test_loss,
                                "test_acc":      self.test_accuracy.item(),
                                "train_acc":     self.train_accuracy.item(),
                                "Batch":         self.batch}

            self.init_state_dict = {'epoch': self.epoch,
                                    'batch': self.batch,
                                    'GenLoss': self.loss_gen.item(),
                                    'DisLoss': self.loss_disc.item(),
                                    'FidScore': self.fid.item(),
                                    'model_state_dict': self.model.state_dict()}
            
            self.test_noise = torch.randn(30, self.noise_dim, device=self.device)
            self.test_labels = torch.tensor([i % 10 for i in range(30)], device=self.device)

        # Saving generated images
        with torch.no_grad():         
            pos_epoch = self.epoch

            generated_images = self._generate(self.test_noise, self.test_labels)
            predictions_gen = self._discriminate(generated_images.detach(), self.test_labels)
            
            self.data_storage.dump_store("generated_images", generated_images)
            self.data_storage.dump_store("epochs_gen", pos_epoch)
            self.data_storage.dump_store("predictions_gen", predictions_gen.detach().cpu().numpy())
            self.data_storage.dump_store("labels_gen", self.test_labels)

    def _save(self):
        torch.save(self.init_state_dict, self.init_save_path)
        
        torch.save({'epoch': self.epoch,
                    'best_values': self.best_values,
                    'best_acc_loss_values': self.other_best_values,
                    'model_state_dict': self.best_state_dict},
                    self.best_save_path)

        torch.save({'epoch': self.epoch,
                    'batch': self.batch,
                    'GenLoss': self.loss_gen.item(),
                    'DisLoss': self.loss_disc.item(),
                    'FidScore': self.fid.item(),
                    'model_state_dict': self.model.state_dict()},
                    self.net_save_path)

        self.parameter_storage.store(self.init_values, "initial_values")
        self.parameter_storage.store(self.best_values, "best_values")
        self.parameter_storage.store(self.other_best_values, "best_acc_loss_values")
        self.parameter_storage.store(self.end_values, "end_values")
        self.parameter_storage.write("\n")
        # if self.best_values["FidScore"] <= 0.8:
        torch.save(self.data_storage, os.path.join(self.result_folder, "data_storage.pt"))
    
    def _load_initial(self):
        checkpoint = torch.load(self.initial_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model.discriminator
    
    def _load_best(self):
        checkpoint = torch.load(self.best_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model.discriminator
    
    def grayscale_to_rgb(self, images):
                # `images` is expected to be of shape [batch_size, 1, height, width]
                return images.repeat(1, 3, 1, 1)

