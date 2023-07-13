

from argparse import ArgumentParser
from your_module_name.utility.experiments import create_experiment_folder, greeting


def main(parser: ArgumentParser = None, **kwargs):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("-name", type=str, default='world', help="Name to greet.")
    
    args = parser.parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    #### Create the experiment folder ####
    create_experiment_folder('hello', args.name, args.__dict__)
    
    #### Run the actual experiment here ####
    greeting(args.name)
    
if __name__ == '__main__':
    main()