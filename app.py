import argparse
from demo.UI import Main_ui

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SEAGULL', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to seagull model', default='Zevin2023/SEAGULL-7B')
    parser.add_argument('--example_path', help='path to examples', default='./imgs/Examples')
    args = parser.parse_args()
    
    demo = Main_ui(args).load_demo()
    demo.launch(server_port=7530)