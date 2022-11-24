from model import build_model
import profit_gym


def train():
    model = build_model(8, 4)
    profit_gym = profit_gym.ProfitGym(width, height, turns, products)
