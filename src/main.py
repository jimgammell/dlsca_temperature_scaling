import argparse

def test():
    from models import test as models_test
    models_test()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test', default=False, action='store_true', help='Run tests to validate dataset and model code.'
    )
    args = parser.parse_args()
    
    if args.test:
        test()

if __name__ == '__main__':
    main()