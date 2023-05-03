from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from data import PTBXLDataModule 

from plmodule import ECGModel

       

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--wandb_name", default="")
        parser.add_argument("--wandb_entity", default="deeplearningecg")
        parser.add_argument("--wandb_project", default="PTBXL_ECG_Dropout")
        
        
        
    def instantiate_classes(self):
        
        # Call to the logger before initiate other clases, because Trainer class init logger if we didn´t do it
        logger = WandbLogger(entity=self.config[self.config.subcommand].wandb_entity, 
                             project=self.config[self.config.subcommand].wandb_project,
                             name=self.config[self.config.subcommand].wandb_name)
        super().instantiate_classes() # call to super class instatiate_classes()


def cli_main():

    cli = MyLightningCLI(ECGModel, PTBXLDataModule,
                        save_config_kwargs={"config_filename": "test_config.yaml", 'overwrite':True})


if __name__ == "__main__":
    cli_main()


    



