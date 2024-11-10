import itertools
import pandas as pd

pd.set_option('display.max_columns', 10, 'display.width', 1000)

DATA_FOLDER = 'data/'
fn_df = DATA_FOLDER + 'Highway-Rail_Grade_Crossing_Accident_Data__Form_57__20240925.csv'
df = pd.read_csv(fn_df)
df['Date'] = pd.to_datetime(df['Date'])
# subject = ["train", "trains", "rail", "rails", "railway", "railways", "amtrak"]
# action = ["accident", "accidents", "accidental", "accidentally", "mishap", "mishaps", "crash", "crashes", "crashed", "crashing", "kill", "kills", "killed", "killing", "derail", "derails", "derailed", "derailing", "derailment", "derailments", "hit", "hits", "hitting", "injure", "injures", "injured", "injuring", "injury", "injuries", "hurt", "hurts", "hurting", "collide", "collides", "collided", "collision", "collisions", "smash", "smashes", "smashed", "smashing", "wreck", "wrecks", "wrecked", "wrecking", "slam", "slams", "slammed", "slamming", "strike", "strikes", "struck", "striking", "ram", "rams", "rammed", "ramming", "die", "died", "dying"]

# noun
subject_noun = ['train']
action_noun = ['crash', 'collision', 'pileup', 'accident', 'smash', 'run into', 'struck', 'hit']

df['City Name'].value_counts()
df['County Name'].value_counts()
df['State Name'].value_counts()

# active
subject_active = ['A train']
action_active = ['crashed into', 'collided with', 'hit', 'smashed into', 'struck', 'ran into']
object_active= ['a car', 'a truck', 'a trailer', 'a man', 'a woman', 'a motorcycle', 'a bicycle', 'a bus', 'a van']
[f'{s} {a} {o}.' for s, a, o in itertools.product(subject_active, action_active, object_active)]

# passive
action_passive = ['killed', 'injured', 'hurt', 'die', 'dead', 'hit']
proposition_passive = ['by', 'as', 'in', 'from', 'on', 'when']
subject_passive = ['train', 'trains']

[f'{s} {a}.' for s, a in itertools.product(subject_passive, action_passive)]
# print(' OR '.join([f'"{s} {a}"' for s, a in itertools.product(subject_noun, action_noun)]))