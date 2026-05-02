from multiagent.custom_scenarios import load


def MPEEnv(args):
    scenario = load(args.scenario_name + ".py").Scenario()
    world = scenario.make_world(args=args)
    if args.algorithm_name in ["mappo", "rmappo"]:
        from multiagent.environment import MultiAgentPPOEnv as MultiAgentEnv
    else:
        from multiagent.environment import MultiAgentOffPolicyEnv as MultiAgentEnv

    return MultiAgentEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback if hasattr(scenario, "info_callback") else None,
        scenario_name=args.scenario_name,
    )


def GraphMPEEnv(args):
    assert "graph" in args.scenario_name, "Only use graph env for graph scenarios"
    scenario = load(args.scenario_name + ".py").Scenario()
    world = scenario.make_world(args=args)
    from multiagent.environment import MultiAgentGraphEnv

    return MultiAgentGraphEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        graph_observation_callback=scenario.graph_observation,
        update_graph=scenario.update_graph,
        id_callback=scenario.get_id,
        info_callback=scenario.info_callback,
        scenario_name=args.scenario_name,
    )
