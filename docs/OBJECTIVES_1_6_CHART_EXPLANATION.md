# Objective 1-6 图表结果说明

本文档用于解释 `outputs/obj1` 到 `outputs/obj5` 中每张图表的含义，以及 Objective 6 在 Streamlit dashboard 中已经实现的内容。它适合用于 PPT 讲解、答辩说明或项目结果汇报。

## Objective 1: Knowledge Base & Keyword Extraction

目标：证明项目已经从 YC、香港公司注册处、商业指南、社交媒体四类数据源中建立了有广度和深度的知识库，并能提取关键词特征。

输出目录：

```text
outputs/obj1/
```

### `yc_top20_tfidf.png`

含义：展示 YC 公司数据中 TF-IDF 权重最高的 20 个关键词。

讲解重点：这些关键词反映 YC startup 数据库中最常见、最能区分主题的创业领域，例如 AI、healthcare、workflow、automation、customer、platform 等。它证明知识库不是只有公司名称，而是包含可用于 startup pattern analysis 的语义内容。

### `hk_registry_top20_tfidf.png`

含义：展示香港公司注册处相关文本中 TF-IDF 权重最高的 20 个关键词。

讲解重点：这些关键词通常围绕 incorporation、company name、business registration、deregistration、BRN、UBI、Companies Registry 等主题。它证明系统收集了香港公司注册与合规流程相关的数据，而不是只依赖通用创业内容。

### `business_guides_top20_tfidf.png`

含义：展示商业运营和创业指南数据中的前 20 个关键词。

讲解重点：这些关键词体现 pricing、inventory、operations、market research、business planning 等实际运营问题。它支持 chatbot 回答创业者在启动和运营阶段遇到的实践问题。

### `social_media_top20_tfidf.png`

含义：展示社交媒体情绪数据中最有代表性的关键词。

讲解重点：这些关键词用于说明系统具备 sentiment / market feedback 数据基础，可以分析用户反馈、正负面情绪和市场讨论趋势。

### `knowledge_base_sunburst.png` / `knowledge_base_sunburst.html`

含义：用 sunburst chart 展示知识库按 source 和 sub-category 的层级分布。

讲解重点：中心到外层展示了数据源结构，例如 YC、HK Registry、Business Guides、Social Media，再展开到具体 topic/category。它证明知识库不是扁平堆叠，而是按照来源和主题组织起来的。

### `data_source_summary.png`

含义：汇总每个数据源的文档数量、token 规模和更新频率等基本信息。

讲解重点：这张表用于回答“你们到底收集了多少数据”。当前数据规模包括 YC 约 5494 行、HK Registry 约 54 行、Business Guides 约 33 行、Social Media 约 732 行。

### `essential_statistics.png`

含义：展示每个数据源的 keyword density、vocabulary size、unigram/bigram coverage、data sparsity 等基础统计指标。

讲解重点：这张表证明每类数据不仅有数量，也有可分析的词汇结构。词汇量和稀疏度可以解释为什么需要向量检索和关键词抽取，而不是只做简单字符串匹配。

### CSV 辅助文件

```text
outputs/obj1/data_source_summary.csv
outputs/obj1/essential_statistics.csv
outputs/obj1/sunburst_source_data.csv
```

含义：这些 CSV 是图表背后的结构化数据，方便在 PPT 或报告中引用具体数字。

## Objective 2: Topic Classification

目标：证明系统能理解用户问题属于哪个主题，例如香港合规、商业运营、YC/startup patterns、社交媒体情绪。

输出目录：

```text
outputs/obj2/
```

当前核心结果：

```text
Test queries: 96
Topic accuracy: 94.79%
Weighted F1: 94.94%
Cohen's Kappa: 93.06%
Misclassification cases: 5
```

### `confusion_matrix_counts.png`

含义：用混淆矩阵展示真实 topic 与预测 topic 的数量关系。

讲解重点：对角线上的数字代表预测正确的数量，非对角线代表误分类。该图用于说明系统大多数时候能正确识别用户问题类型，也能清楚展示少数混淆发生在哪里。

### `confusion_matrix_normalized.png`

含义：标准化后的混淆矩阵，用百分比展示每个真实类别被预测到各类别的比例。

讲解重点：这张图比 count matrix 更适合比较不同类别的表现，因为它消除了不同类别样本数量带来的影响。

### `query_tsne_topic_map.png`

含义：将 query 的 TF-IDF / embedding 特征降维到二维空间，并按预测 topic 着色。

讲解重点：两个 dimension 没有直接业务含义，它们是 t-SNE 降维得到的视觉坐标。图中点越聚集，说明同类问题在语义空间中越相似；不同颜色分开，说明分类边界较清晰。

### `classification_report_table.png`

含义：展示每个 topic 的 precision、recall、F1-score 和 support。

讲解重点：precision 表示预测为该类时有多少是真的，recall 表示该类问题有多少被找回，F1 是综合指标。它用于证明每个主题不是只靠总体准确率支撑。

### `topic_metrics_bar.png`

含义：用柱状图比较各 topic 的 precision、recall 和 F1。

讲解重点：这张图适合在 PPT 中快速展示哪个类别表现最好、哪个类别相对更容易混淆。

### `topic_distribution.png`

含义：展示测试集中各 topic 的问题数量分布。

讲解重点：它证明 evaluation set 覆盖多个主题，而不是只测试一种问题。

### `objective2_statistics_table.png`

含义：汇总 Objective 2 的关键指标，例如 test query count、accuracy、weighted F1、Cohen's Kappa、misclassification count。

讲解重点：这是 Objective 2 的总览表，适合放在结论页。

### `topic_misclassification_cases.csv`

含义：列出被误分类的问题。

讲解重点：用于解释模型边界，例如某些问题同时包含 sentiment 和 startup/healthcare 词汇，导致分类不完全明确。它证明评估不是全部人为设计成完美结果，而是保留了真实边界情况。

## Objective 3: Information Retrieval & Ranking

目标：证明 chatbot 的检索功能可以把相关知识 chunk 排在前面。

输出目录：

```text
outputs/obj3/
```

当前核心结果：

```text
Source Recall@1: 94.79%
Source Recall@3: 97.92%
Source Recall@5: 97.92%
Relevant Recall@1: 69.44%
Relevant Recall@3: 80.56%
Relevant Recall@5: 81.94%
Relevant MRR: 0.7465
Relevant NDCG@5: 0.7651
Average latency: 178.95 ms
```

### `search_relevance_histogram.png`

含义：展示 query 与检索结果之间的相似度分数分布。

讲解重点：如果大部分 top results 的分数较高，说明检索器能找到与问题语义相关的 chunks。如果分布有长尾，也能说明某些问题更难检索。

### `recall_at_k_curve.png`

含义：展示 Recall@1、Recall@3、Recall@5 的变化曲线。

讲解重点：它回答“正确信息是否出现在前 K 个结果中”。曲线从 @1 到 @5 上升，说明给模型更多 top chunks 后，正确答案出现概率提高。

### `retrieval_metrics_table.png`

含义：汇总 MRR、NDCG、Recall@K、latency 等检索指标。

讲解重点：MRR 反映第一个正确结果平均排得多靠前；NDCG@5 反映前 5 个结果的排序质量；latency 说明检索速度。

### `topk_retrieval_samples_table.png`

含义：展示用户问题与 top retrieved snippets 的对应关系。

讲解重点：这张表适合做 qualitative evidence，说明系统不只是指标高，实际检索出来的片段也和问题相关。

### `relevant_rank_distribution.png`

含义：展示相关结果通常出现在第几名。

讲解重点：如果相关结果多出现在 rank 1 或 rank 2，说明排序器有能力把高价值内容放在前面。

### `source_recall_by_family.png`

含义：按数据源类型展示 source recall 表现。

讲解重点：它说明系统不是只对 YC 好用，也能对 HK official、business guide、sentiment 等不同来源做检索。

### CSV 辅助文件

```text
outputs/obj3/retrieval_score_distribution.csv
outputs/obj3/source_recall_by_family.csv
```

含义：保存检索分数分布和按来源统计的 recall 数据，可用于报告中引用具体数值。

## Objective 4: Answer Summarization

目标：证明 chatbot 能把检索到的内容组织成简洁、有来源依据的答案。

输出目录：

```text
outputs/obj4/
```

当前核心结果：

```text
Generated answers: 96
ROUGE-1: 0.1209
ROUGE-2: 0.0238
ROUGE-L: 0.0878
Source attribution counts:
- YC: 149
- Business Guide: 70
- HK Official: 69
```

### `rouge_scores_bar.png`

含义：展示生成答案与 gold answer 之间的 ROUGE-1、ROUGE-2、ROUGE-L 分数。

讲解重点：ROUGE 衡量生成答案和参考答案之间的词汇重合度。由于 chatbot 是 RAG 生成式回答，不一定逐字复述 gold answer，因此分数不需要接近 1；它更多用于证明答案覆盖了部分关键内容。

### `source_attribution_pie.png`

含义：展示生成答案所引用或依赖的来源分布。

讲解重点：它说明答案不是凭空生成，而是来自 YC、HK official、business guide 等检索来源。当前 YC 引用最多，HK official 和 business guide 也有明显贡献。

### `answer_length_comparison.png`

含义：比较 raw retrieved text 和 generated answer 的长度。

讲解重点：它证明 summarization 有压缩作用，chatbot 会把较长、噪声较多的检索内容变成更短的用户可读答案。

### `rouge_by_topic_group.png`

含义：按 topic group 展示 ROUGE 分数差异。

讲解重点：它可以说明系统在哪些主题上总结效果更好，哪些主题更难。例如合规问题可能更依赖官方 wording，startup example 问题可能答案表达更开放。

### `answer_evaluation_summary_table.png`

含义：汇总 Objective 4 的关键评估结果，例如答案数量、ROUGE 分数、source attribution 等。

讲解重点：这是 Objective 4 的结果总览，适合放在 PPT 的 summary slide。

### `retrieved_vs_answer_table.png`

含义：展示一个或两个样例中，用户 query、top retrieved source、generated answer、gold answer 的对比。

讲解重点：这张图用于 qualitative demonstration，说明检索内容如何被压缩、整理并转化为最终答案。

### CSV 辅助文件

```text
outputs/obj4/retrieved_vs_answer_full.csv
outputs/obj4/answer_length_stats.csv
outputs/obj4/rouge_by_topic_group.csv
outputs/obj4/source_attribution_counts.csv
```

含义：保存完整样例和统计结果，可用于进一步筛选 PPT 中要展示的案例。

## Objective 5: Follow-Up Question Prediction

目标：证明 chatbot 能根据用户当前问题和答案，预测创业者下一步可能关心的问题，并通过点击行为验证这些推荐是否有用。

输出目录：

```text
outputs/obj5/
```

当前核心结果：

```text
Events: 78
Suggestions shown: 63
Suggestions clicked: 15
Overall CTR: 23.81%
Turns with suggestions: 21
Turns with click: 15
Turn-level prediction accuracy: 71.43%
Rank-choice perplexity proxy: 2.4679
```

说明：Objective 5 的补充数据中，seeded evaluation events 使用字段 `"data_source": "seeded_objective5_evaluation"` 标记，用于覆盖 HK compliance、business operations、YC/funding、social sentiment 四类路径。

### `user_journey_sankey.png`

含义：展示用户从当前问题进入下一步 follow-up 的 journey flow。

讲解重点：例如 Start -> Registering a Business -> Registering a Business，Start -> Seeking Funding -> Seeking Funding，Start -> Market Sentiment -> Market Sentiment。它证明 chatbot 的 follow-up 不是孤立问题，而是可以形成创业者 journey。

### `prediction_accuracy_bar.png`

含义：展示 turn-level prediction accuracy 和不同推荐位置的 CTR。

讲解重点：Any suggestion clicked 表示只要用户点击了三个推荐问题中的任意一个，就认为该 turn 的预测命中。Suggestion 1/2/3 CTR 展示不同推荐排序位置的点击效果。

### `ctr_by_topic.png`

含义：按 topic 展示 follow-up suggestion 的点击率。

讲解重点：它可以说明在哪些领域 follow-up 更容易被用户接受。例如 HK compliance、YC funding、social sentiment、business operations 都有覆盖，因此不是单一领域的结果。

### `clicked_rank_distribution.png`

含义：展示用户点击的是第 1、第 2 还是第 3 个推荐问题。

讲解重点：它用于分析推荐排序是否合理。如果 rank 1 和 rank 2 点击较多，说明前两个推荐通常更贴近用户下一步需求。

### `followup_metrics_table.png`

含义：Objective 5 的指标总览表。

讲解重点：适合放在 PPT 中作为 summary，展示 shown、clicked、CTR、turn-level accuracy、rank-choice perplexity proxy 等核心数字。

### CSV 辅助文件

```text
outputs/obj5/followup_shown_events.csv
outputs/obj5/followup_clicked_events.csv
outputs/obj5/journey_stage_edges.csv
outputs/obj5/ctr_by_topic.csv
```

含义：保存 follow-up 展示事件、点击事件、journey edge 和 topic CTR 的结构化数据。

### 关于 perplexity 的说明

当前实现的是 rank-choice perplexity proxy，不是 token-level language model perplexity。原因是当前 Ollama follow-up generation 调用没有返回 token log probabilities。

讲解时建议表述为：系统使用点击 rank 分布计算 recommendation choice perplexity，用来衡量用户点击行为集中还是分散。真正的 LLM token-level perplexity 需要模型提供 logprob 支持。

## Objective 6: Integrated Dashboard & Chatbot Analytics

目标：为项目 stakeholders 提供一个 command center，用于查看系统数据覆盖、评估指标、情绪分析、follow-up 表现和当前会话状态。

Objective 6 主要在 Streamlit dashboard 中实现，而不是全部导出为静态 PNG。

运行方式：

```cmd
streamlit run app.py
```

进入页面后打开：

```text
Analytics Dashboard
```

### Overview Metrics

位置：Dashboard 顶部。

实现内容：

- Companies
- Indexed Chunks
- Topics
- Sentiment Rows
- Chat Turns

含义：这是系统总体数据规模和当前会话状态的 dashboard pulse。它说明知识库、情绪数据和当前聊天 session 是否已经加载并产生交互。

### Evaluation Results

实现内容：

- Topic Accuracy
- Weighted F1
- Recall@5
- MRR
- Avg Latency
- Evaluation metric bar chart
- Classification report table
- Top-K retrieval samples table

含义：这是系统质量和性能的集中视图。它把 Obj2、Obj3、Obj4 的关键离线评估指标放到 dashboard 中，便于 stakeholder 不打开每个输出文件也能看到系统表现。

### Follow-Up Prediction Analytics

实现内容：

- Suggestions Shown
- Suggestions Clicked
- Follow-Up CTR
- Logged Events
- CTR by suggestion rank
- clicked follow-up journey table

含义：这是 Objective 5 在 dashboard 中的实时/累计分析面板。它证明系统能记录推荐问题展示和点击行为，并用 CTR 衡量 follow-up prediction 的有效性。

### Knowledge Source Distribution

实现内容：donut chart 展示知识库不同 source family 的占比。

含义：它说明系统回答问题时可用的数据来源结构，例如 YC、HK official、business guide 等，避免知识库被误解为单一来源。

### Top Knowledge-Base Topics

实现内容：bar chart 展示 YC knowledge base 中最常见的 topic labels。

含义：它相当于 dashboard 中的 interactive topic map 简化版本，用于展示当前知识库覆盖了哪些创业主题。

### Sentiment Label Distribution

实现内容：donut chart 展示 sentiment dataset 中不同情绪标签的数量分布。

含义：它说明 sentiment 数据集的整体情绪构成，例如 neutral-heavy 或 positive/negative 的比例。

### Platform by Sentiment

实现内容：按 platform 展示不同 sentiment label 的分布。

含义：它用于比较 Twitter、Instagram、Facebook 等平台上的情绪差异，属于 Objective 6 中 sentiment heatmap / sentiment trend 的 dashboard 实现版本。

### Average Engagement by Platform

实现内容：scatter chart 和 dataframe 展示不同平台的 average likes / retweets。

含义：它把 sentiment 分析和 engagement 信号结合起来，帮助观察哪些平台上的讨论更活跃。

### Current Session Retrieval Mix

实现内容：当前聊天 session 中 retrieved sources 的来源分布图。

含义：这是实时会话层面的可解释性图表。它说明当前回答主要依赖了哪些 source family，例如 YC、HK official 或 business guide。

### Top Positive Keywords / Top Neutral Keywords

实现内容：bar chart 展示情绪数据中的高频关键词。

含义：用于解释不同情绪类别背后的主要文本特征，帮助理解 sentiment analysis 的内容来源。

### Key Insights

实现内容：自动生成文字洞察。

含义：它把 source coverage、sentiment limitation、knowledge base scale、live session source mix 等信息转成 stakeholder 更容易理解的结论。

### Diagnostic Notes

实现内容：dashboard 底部的系统说明和限制提醒。

含义：用于明确当前系统的注意事项，例如 YC 数据较多、sentiment 数据偏 neutral、current session retrieval mix 可解释回答来源。

### Objective 6 当前未完整实现的部分

以下内容已有部分替代指标，但还不是完整长期监控系统：

- API uptime：目前没有持续 uptime log，只能通过 app health 和运行状态间接说明。
- Model confidence gauge：目前使用 retrieval score、Recall@K、MRR、topic accuracy 等间接指标，还没有真正的实时 confidence gauge。
- Real-time active user sessions：当前 dashboard 展示当前 session 的 Chat Turns 和 retrieval mix，但没有多用户并发 session 监控。

### Objective 6 静态输出

当前已有：

```text
outputs/obj6/Real-Time Pulse.png
```

其余 Objective 6 内容主要通过 Streamlit dashboard 动态展示。

## 总体说明

Objective 1-5 已经形成静态图表和 CSV 证据文件，适合直接放入 PPT。Objective 6 主要体现为运行中的 dashboard，用于展示项目的综合监控能力。若需要完整 PPT 截图，可以运行 Streamlit app 后对 Analytics Dashboard 截屏，或者后续再补充 `generate_obj6_figures.py` 将 dashboard 关键模块导出为静态 PNG。
