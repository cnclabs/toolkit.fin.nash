export type NumericMention = {
  text: string;
  value: number;
  raw?: string;
  normalized_value?: number;
  start: number;
  end: number;
  unit?: string;
  suffix?: string;
};

export type NumericAlignment = {
  source_index: number;
  target_index: number;
  source_text: string;
  target_text: string;
  contextual_similarity: number;
  magnitude_similarity: number;
  valid: boolean;
};

export type PairwiseNumeric = {
  source_index: number;
  target_index: number;
  source_text: string;
  target_text: string;
  source_value: number;
  target_value: number;
  absolute_difference: number;
  normalization_factor: number;
  contextual_similarity: number;
  pairwise_numeric_similarity: number;
  valid: boolean;
};

export type NashTrace = {
  metadata?: Record<string, unknown>;
  input: {
    sentence1: string;
    sentence2: string;
  };
  numeric_extraction: {
    sentence1_numbers: NumericMention[];
    sentence2_numbers: NumericMention[];
  };
  masking: {
    masked_sentence1: string;
    masked_sentence2: string;
  };
  tokenization: {
    sentence1_tokens: string[];
    sentence2_tokens: string[];
    number_token_spans_sentence1: number[][];
    number_token_spans_sentence2: number[][];
  };
  textual_similarity: {
    score: number;
    backend: string;
  };
  numeric_alignment: {
    threshold: number;
    similarity_matrix: number[][];
    alignments_s1_to_s2: NumericAlignment[];
    alignments_s2_to_s1: NumericAlignment[];
    valid_alignments_s1_to_s2: NumericAlignment[];
    valid_alignments_s2_to_s1: NumericAlignment[];
  };
  numeric_similarity: {
    pairwise_s1_to_s2: PairwiseNumeric[];
    pairwise_s2_to_s1: PairwiseNumeric[];
    directional_s1_to_s2: number;
    directional_s2_to_s1: number;
    score: number;
  };
  aggregation: {
    idf_text_mass: number;
    idf_numeric_mass: number;
    final_score: number;
  };
  baseline_comparison: {
    baseline_score: number;
    nash_score: number;
    delta: number;
  };
};

export type DemoExample = {
  id: string;
  label: string;
  sentence1: string;
  sentence2: string;
  trace_file: string;
};

export type ModelOption = {
  id: string;
  label: string;
  backend: string;
  requires_model_dependencies: boolean;
};

export type BaselineComparisonRow = {
  baseline_id: string;
  baseline_label: string;
  backend: string;
  baseline_score: number;
  nash_score: number;
  numeric_score: number;
  text_score: number;
  source: string;
};

export type BaselineComparisonResponse = {
  input: {
    sentence1: string;
    sentence2: string;
  };
  rows: BaselineComparisonRow[];
  traces: NashTrace[];
  selected_trace: NashTrace | null;
};

export type BatchPair = {
  sentence1: string;
  sentence2: string;
};

export type BatchRow = BatchPair & {
  index: number;
  baseline_score: number;
  nash_score: number;
  numeric_score: number;
  text_score: number;
  delta: number;
};

export type BatchResponse = {
  baseline: ModelOption;
  summary: {
    n_pairs: number;
    average_baseline_score: number;
    average_nash_score: number;
    average_numeric_score: number;
    strong_disagreement_count: number;
  };
  rows: BatchRow[];
};

export type BatchPreset = {
  id: string;
  label: string;
  description: string;
  pairs: BatchPair[];
};

export type TripletResponse = {
  input: {
    anchor: string;
    positive: string;
    negative: string;
  };
  rows: Array<{
    baseline_id: string;
    baseline_label: string;
    backend: string;
    baseline_positive_score: number;
    baseline_negative_score: number;
    nash_positive_score: number;
    nash_negative_score: number;
    baseline_correct: boolean;
    nash_correct: boolean;
  }>;
  traces: Array<{
    baseline_id: string;
    positive: NashTrace;
    negative: NashTrace;
  }>;
};

export type SensitivityResponse = {
  baseline: ModelOption;
  anchor: string;
  rows: Array<{
    variant: string;
    baseline_score: number;
    nash_score: number;
    numeric_score: number;
    text_score: number;
  }>;
};

export type HistoryItem = {
  id: string;
  preview: string;
  baseline: string;
  baseline_score: number;
  nash_score: number;
  text_score?: number;
  numeric_score?: number;
  trace?: NashTrace | null;
  comparison?: BaselineComparisonResponse | null;
  timestamp: string;
};

export type DashboardSession = {
  schema: "nash-eval-session";
  version: 1;
  exported_at: string;
  mode: "static" | "live";
  selected_example_id: string;
  sentence1: string;
  sentence2: string;
  selected_baseline_id: string;
  trace: NashTrace | null;
  comparison: BaselineComparisonResponse | null;
  batch_result: BatchResponse | null;
  triplet_result: TripletResponse | null;
  sensitivity_result: SensitivityResponse | null;
  history: HistoryItem[];
};

export type DemoBaseline = {
  name: string;
  baseline_score: number;
  nash_score: number;
  text_score: number;
  numeric_score: number;
};

export type DemoNumber = {
  id: string;
  text: string;
  value: number;
  start: number;
  end: number;
};

export type DemoAlignment = {
  source: string;
  target: string | null;
  source_text: string;
  target_text: string | null;
  alignment_similarity: number;
  numeric_gap: number | null;
  status: "aligned" | "unmatched" | "filtered";
};

export type DemoHeatmap = {
  rows: string[];
  columns: string[];
  values: number[][];
};

export type DemoPairTrace = {
  id: string;
  type: "pairwise";
  title: string;
  sentence1: string;
  sentence2: string;
  baselines: DemoBaseline[];
  numbers1: DemoNumber[];
  numbers2: DemoNumber[];
  alignments: DemoAlignment[];
  heatmap: DemoHeatmap;
  masked_sentence1: string;
  masked_sentence2: string;
};

export type DemoTriplet = {
  id: string;
  type: "triplet";
  title: string;
  anchor: string;
  positive: string;
  negative: string;
  baselines: Array<{
    name: string;
    baseline_anchor_positive: number;
    baseline_anchor_negative: number;
    nash_anchor_positive: number;
    nash_anchor_negative: number;
    baseline_ranking: "correct" | "wrong";
    nash_ranking: "correct" | "wrong";
  }>;
};
