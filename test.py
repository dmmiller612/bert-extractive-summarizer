# -*- coding: gbk -*-
from transformers import *

# Load model, model config and tokenizer via Transformers
custom_config = AutoConfig.from_pretrained('bert-base-chinese')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
custom_model = AutoModel.from_pretrained('bert-base-chinese', config=custom_config)

from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer

body = '中国基金报记者 吴君 n今年公募行业再度掀起离职潮，董承非、周应波、赵诣、崔莹等明星基金经理离职，他们原来管理的明星绩优产品由别的基金经理接管，投资者该如何看待，是去是留？另外，明星基金经理的离开，对基金公司带来哪些影响，人才是否能接续上，也备受业内关注。记者采访了一些业内人士来解答这些问题，他们认为投资者可以给予一定的观察期，来考察新任基金经理，再做申赎的打算。人才流动是基金行业正常现象，新任者也能延续辉煌，基金公司需要加强人才梯队建设。 n离职潮后明星产品接任者受关注投资者不妨留观察期、重新审视 nWind数据显示，截至4月17日，年内共有64家基金公司的92名基金经理离任，相比去年同期增长23%。 n部分明星基金经理的离职引发关注，比如兴证全球基金原副总经理董承非此前官宣离职，他原来管理的兴全趋势等基金由谢治宇、董理等基金经理接管。中欧基金原明星基金经理周应波卸任后，中欧时代先锋等基金由周蔚文、刘伟伟等接手。还有，农银汇理基金赵诣、宝盈基金肖肖离任后，公司安排了不同基金经理接手其原来管理的明星基金产品。 n贝塔研究院认为，不少投资者都是看基金经理来“下单”，正所谓“选基，就是选人”。今年离职的基金经理中，投资年限超过5年、投资生涯年化回报率超过10%的基金经理共有12位。“离职基金经理管理的明星产品由公司安排其他人接任，主要影响在于基金经理之间的投资理念、擅长领域、投资风格不尽相同，基金经理的变动直接影响着基金的业绩表现，对于明星产品，大多数基民核心是看基金经理来购买基金，基金经理的离职可能对部分投资者继续持有明星产品的态度有所松动。” n贝塔研究院建议，持有主动权益类基金的投资者，可以追踪新任基金经理的履历、历史业绩、历史口碑、投资理念等情况，综合进行考量，给可以予自己一定时间的“观察期”，来考察在新任基金经理管理下的基金产品是否符合自身预期，再做持有或赎回打算。 n基煜研究称，基金经理离职之后，新任者的操作思路相较于之前是有所差异的，即使大类风格延续，比如依然是科技或消费风格，但是基金经理的投资方法一般都会有一定差异。所以建议投资者重新审视该类产品，避免简单的以业绩线性外推。“单以基金经理变更来决定是否申赎，这种行为本身不够客观。建议投资者面临这种情况时，重新审视新任基金经理的投资理念，考察其管理水平，再决定申赎。” n通联数据高级基金研究员沈秋敏表示，主	'
# model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
# a =model(body, min_length =5,ratio=0.1,num_sentences=2)
# print(a)


from summarizer.sbert import SBertSummarizer

model = SBertSummarizer('distiluse-base-multilingual-cased-v1')
result = model(body, min_length=5, num_sentences=3)
print(result)