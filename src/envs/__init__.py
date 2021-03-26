from gym.envs.registration import register

# ----------------------------------------- Half-Cheetah

register(
    id='Half-Cheetah-RM1-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM11',
    max_episode_steps=1000,
)
register(
    id='Half-Cheetah-RM2-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM12',
    max_episode_steps=1000,
)



# ----------------------------------------- WATER
for i in range(11):
    w_id = 'Water-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

for i in range(11):
    w_id = 'Water-single-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRM10EnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

# ----------------------------------------- OFFICE
register(
    id='Office-v0',
    entry_point='envs.grids.grid_environment:OfficeRMEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-v0',
    entry_point='envs.grids.grid_environment:OfficeRM3Env',
    max_episode_steps=1000
)

register(
    id='Office-noisy-v0',
    entry_point='envs.grids.grid_environment:OfficeNoisyRM3Env',
    max_episode_steps=1000
)

# ----------------------------------------- CRAFT
for i in range(11):
    w_id = 'Craft-M%d-v0'%i
    w_en = 'envs.grids.grid_environment:CraftRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )

for i in range(11):
    w_id = 'Craft-single-M%d-v0'%i
    w_en = 'envs.grids.grid_environment:CraftRM10EnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=1000
    )

 # --- GARDEN / HARVESTING
register(
    id='Garden-v0',
    entry_point='envs.grids.garden_environment:GardenEnvNoSlip',
    max_episode_steps=250
)

register(
    id='Harvest-v0',
    entry_point='envs.grids.harvest_environment:HarvestRMEnv',
    max_episode_steps=250
)

register(
    id='GardenS-v0',
    entry_point='envs.grids.garden_environment:GardenEnvSlip5',
    max_episode_steps=250
)

# --- MINING

for i in range(1, 5):
    register(
        id=f'MiningT{i}-v0',
        entry_point=f'envs.grids.grid_environment:MiningRMEnvT{i}',
        max_episode_steps=1000
    )
    register(
        id=f'MiningST{i}-v0',
        entry_point=f'envs.grids.grid_environment:MiningRMEnvST{i}',
        max_episode_steps=1000
    )

register(
    id=f'MiningT5-v0',
    entry_point='envs.grids.grid_environment:MiningRMEnvT5',
    max_episode_steps=1000
)
