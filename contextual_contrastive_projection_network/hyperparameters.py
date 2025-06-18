# hyperparameters.py
from dataclasses import dataclass


@dataclass
class HyperParameters:
	input_layer: int = 1024
	hidden_layer: int = 500
	output_layer: int = 500
	is_hidden_layer: bool = True

	activation: str = "silu"
	noise: str = "dropout"
	dropout: float = 0.1
	enable_gate: bool = True

	batch_size: int = 50
	epochs: int = 30
	lr: float = 4e-6
	loss_fn: str = "triplet_loss"
	triplet_loss_margin: float = 0.5
	hard_negative_ratio: float = 0.1
	instances_per_type: int = 25


class TrainTypes:
	types = [
		'education',
		'airport',
		'restaurant',
		'sportsleague',
		'disease',
		'hospital',
		'painting',
		'other',
		'library',
		'sportsevent',
		'soldier',
		'game',
		'educationaldegree',
		'broadcastprogram',
		'mountain',
		'road/railway/highway/transit',
		'company',
		'politician',
		'attack/battle/war/militaryconflict',
		'astronomything',
		'language',
		'train',
		'scholar',
		'bodiesofwater',
		'chemicalthing',
		'director',
		'showorganization',
		'writtenart',
		'disaster',
		'medical',
		'music',
		'airplane',
		'biologything',
		'theater',
		'sportsteam',
		'government/governmentagency',
		'livingthing',
		'artist/author',
		'protest',
		'god'
	]


class TestTypes:
	types =  [
	'island',
	'athlete',
	'politicalparty',
	'actor',
	'software',
	'sportsfacility',
	'weapon',
	'food',
	'election',
	'car',
	'currency',
	'park',
	'award',
	'GPE',
	'media/newspaper',
	'law',
	'religion',
	'film',
	'hotel',
	'ship'
]