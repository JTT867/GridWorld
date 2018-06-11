from environment import Env
from New_QLearning import QLearning


if __name__=="__main__":
    env = Env()
    QL = QLearning(list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        while True:
            env.render()

            # take action and proceed one step in the environment
            action = QL.get_action(str(state))
            next_state, reward, done = env.step(action)

            # with sample <s,a,r,s'>, agent learns new q function
            QL.learn(str(state), action, reward, str(next_state))

            state = next_state
            env.print_value_all(QL.q_table)

            # if episode ends, then break
            if done:
                break
