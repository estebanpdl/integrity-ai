# -*- coding: utf-8 -*-

# import modules
from runners.blueprint_runner import handle_blueprint
from runners.claim_analysis_runner import handle_claim_analysis
from runners.evaluation_runner import handle_evaluation

# maps commands to respective runners/handlers
command_dispatcher = {
    'blueprint': handle_blueprint,
    'claim-analysis': handle_claim_analysis,
    'evaluation': handle_evaluation
}
