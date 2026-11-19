import { memo, useEffect, useMemo, useState } from 'react'
import {
  clearAnalysis,
  loadAnalysis,
  saveAnalysis,
} from './analysisStorage'
import type { ReactNode, FormEvent } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  ChartNoAxesCombined,
  ChevronRight,
  CloudUpload,
  Database,
  FileSpreadsheet,
  Layers3,
  LoaderCircle,
  MessageCircle,
  Send,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Trash2,
  X,
} from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './index.css'

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? ''
).replace(/\/$/, '')

type HistoricalPoint = {
  date: string
  nav: number
}

type ChartPoint = {
  date: string
  nav: number
}

type FundMetrics = {
  observations: number
  start_date: string
  end_date: string
  starting_nav: number
  latest_nav: number
  total_return_pct: number
  annualised_return_pct: number
  annualised_volatility_pct: number
  max_drawdown_pct: number
  return_30_observations_pct: number | null
  return_90_observations_pct: number | null
  return_252_observations_pct: number | null
}

type CleaningStats = {
  original_rows: number
  final_rows: number
  removed_rows: number
  invalid_date_rows: number
  invalid_nav_rows: number
  non_positive_nav_rows: number
  duplicate_date_rows: number
}

type FundResult = {
  fund_name: string
  source_file: string
  historical: HistoricalPoint[]
  metrics: FundMetrics
  warnings: string[]
  cleaning: CleaningStats
}

type PreparedFund = FundResult & {
  chartData: ChartPoint[]
}

type ApiResponse = {
  application?: string
  product_name?: string | null
  fund_count?: number
  funds?: FundResult[]
  file_errors?: Array<{
    filename?: string
    error?: string
  }>
  disclaimer?: string
}

type AssistantMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
}

type AssistantApiResponse = {
  reply?: string
  answer?: string
  message?: string
  response?: string
  detail?: string | {
    message?: string
  }
}

const MAX_CHART_POINTS = 1000
const ASSISTANT_CHAT_STORAGE_KEY = 'nav-fund-analytics-assistant-chat'

function loadAssistantMessages(): AssistantMessage[] {
  try {
    const saved = window.localStorage.getItem(ASSISTANT_CHAT_STORAGE_KEY)
    if (!saved) return []

    const parsed: unknown = JSON.parse(saved)
    if (!Array.isArray(parsed)) return []

    return parsed.filter(
      (message): message is AssistantMessage =>
        message !== null &&
        typeof message === 'object' &&
        typeof message.id === 'string' &&
        (message.role === 'user' || message.role === 'assistant') &&
        typeof message.content === 'string',
    )
  } catch {
    return []
  }
}

function prepareChartData(
  historical: HistoricalPoint[],
): ChartPoint[] {
  const validPoints = historical
    .map((point) => ({
      date: String(point.date ?? ''),
      nav: Number(point.nav),
    }))
    .filter(
      (point) =>
        point.date.length > 0 &&
        Number.isFinite(point.nav),
    )

  if (validPoints.length <= MAX_CHART_POINTS) {
    return validPoints
  }

  const first = validPoints[0]
  const last = validPoints[validPoints.length - 1]

  const bucketCount = Math.floor(
    (MAX_CHART_POINTS - 2) / 2,
  )

  const bucketSize = Math.ceil(
    (validPoints.length - 2) / bucketCount,
  )

  const sampled: ChartPoint[] = [first]

  for (
    let start = 1;
    start < validPoints.length - 1;
    start += bucketSize
  ) {
    const end = Math.min(
      start + bucketSize,
      validPoints.length - 1,
    )

    let minimumIndex = start
    let maximumIndex = start

    for (let index = start + 1; index < end; index += 1) {
      if (validPoints[index].nav < validPoints[minimumIndex].nav) {
        minimumIndex = index
      }

      if (validPoints[index].nav > validPoints[maximumIndex].nav) {
        maximumIndex = index
      }
    }

    if (minimumIndex < maximumIndex) {
      sampled.push(validPoints[minimumIndex])
      sampled.push(validPoints[maximumIndex])
    } else if (maximumIndex < minimumIndex) {
      sampled.push(validPoints[maximumIndex])
      sampled.push(validPoints[minimumIndex])
    } else {
      sampled.push(validPoints[minimumIndex])
    }
  }

  sampled.push(last)

  return sampled
}

function formatNumber(
  value: number | null | undefined,
  decimals = 2,
): string {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(value)
  ) {
    return 'Not available'
  }

  return value.toFixed(decimals)
}

function formatPercent(
  value: number | null | undefined,
): string {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(value)
  ) {
    return 'Not available'
  }

  return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`
}

function formatDate(dateString: string): string {
  if (!dateString) return '—'

  const [year, month, day] = dateString.split('-').map(Number)

  if (!year || !month || !day) return dateString

  return new Date(year, month - 1, day).toLocaleDateString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })
}

/* =========================================
   ASSISTANT MARKDOWN RENDERER
========================================= */

// Converts HTML break tags into normal Markdown line breaks.
// This prevents literal <br> tags from cluttering assistant replies.
function normaliseAssistantMarkdown(content: string): string {
  return content
    .replace(/<br\s*\/?>/gi, '  \n')
    .replace(/\r\n/g, '\n')
    .trim()
}

function AssistantMarkdown({
  content,
}: {
  content: string
}) {
  const markdown = normaliseAssistantMarkdown(content)

  return (
    <div className="assistant-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="assistant-table-wrapper">
              <table>{children}</table>
            </div>
          ),
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
            >
              {children}
            </a>
          ),
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  )
}

/* =========================================
   DASHBOARD COMPONENTS
========================================= */

function MetricCard({
  label,
  value,
  description,
  icon,
}: {
  label: string
  value: string
  description: string
  icon: ReactNode
}) {
  return (
    <div className="metric-card">
      <div className="metric-card-top">
        <span>{label}</span>
        <div className="metric-icon">{icon}</div>
      </div>

      <strong className="metric-value">{value}</strong>
      <span className="metric-description">{description}</span>
    </div>
  )
}

function ReturnCard({
  label,
  value,
  description,
}: {
  label: string
  value: number | null
  description: string
}) {
  const available =
    value !== null && Number.isFinite(value)

  const valueClass =
    available && value > 0
      ? 'positive-value'
      : available && value < 0
        ? 'negative-value'
        : ''

  return (
    <div className="return-card">
      <span className="return-label">{label}</span>
      <strong className={valueClass}>
        {formatPercent(value)}
      </strong>
      <small>{description}</small>
    </div>
  )
}

const FundDashboard = memo(function FundDashboard({
  fund,
}: {
  fund: PreparedFund
}) {
  const metrics = fund.metrics
  const warnings = fund.warnings ?? []
  const cleaning = fund.cleaning
  const chartData = fund.chartData

  const latestIsPositive = metrics.total_return_pct >= 0

  return (
    <article className="fund-dashboard">
      <div className="fund-dashboard-header">
        <div className="fund-heading-main">
          <div className="fund-number">FUND ANALYSIS</div>

          <h3>{fund.fund_name}</h3>

          <p>
            <FileSpreadsheet size={13} />
            {fund.source_file}
          </p>
        </div>

        <div className="fund-observation-badge">
          <Database size={15} />
          {metrics.observations} observations
        </div>
      </div>

      <div className="metrics-grid">
        <MetricCard
          label="Latest NAV"
          value={formatNumber(metrics.latest_nav, 2)}
          description={`As of ${formatDate(metrics.end_date)}`}
          icon={<Activity size={17} />}
        />

        <MetricCard
          label="Starting NAV"
          value={formatNumber(metrics.starting_nav, 2)}
          description={`Since ${formatDate(metrics.start_date)}`}
          icon={<Layers3 size={17} />}
        />

        <MetricCard
          label="Total return"
          value={formatPercent(metrics.total_return_pct)}
          description="Return across the full period"
          icon={
            latestIsPositive
              ? <TrendingUp size={17} />
              : <TrendingDown size={17} />
          }
        />

        <MetricCard
          label="Annualised return"
          value={formatPercent(metrics.annualised_return_pct)}
          description="Annualised historical return"
          icon={<ArrowUpRight size={17} />}
        />

        <MetricCard
          label="Annualised volatility"
          value={formatPercent(metrics.annualised_volatility_pct)}
          description="Historical NAV return variability"
          icon={<Activity size={17} />}
        />

        <MetricCard
          label="Maximum drawdown"
          value={formatPercent(metrics.max_drawdown_pct)}
          description="Largest peak-to-trough decline"
          icon={<TrendingDown size={17} />}
        />
      </div>

      <section className="fund-chart-section">
        <div className="dashboard-section-heading">
          <div>
            <span className="eyebrow">NAV HISTORY</span>
            <h4>Historical NAV movement</h4>
            <p>
              Daily NAV observations across the available date range.
            </p>
          </div>

          <div className="chart-legend">
            <span className="legend-dot" />
            Historical NAV
          </div>
        </div>

        {chartData.length > 0 ? (
          <>
            <div className="nav-chart">
              <ResponsiveContainer
                width="100%"
                height={320}
                minWidth={0}
              >
                <LineChart
                  data={chartData}
                  margin={{
                    top: 12,
                    right: 18,
                    left: 5,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid
                    stroke="rgba(255,255,255,0.10)"
                    strokeDasharray="4 5"
                    vertical={false}
                  />

                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) =>
                      formatDate(String(value))
                    }
                    tick={{
                      fill: '#aeb4c5',
                      fontSize: 10,
                    }}
                    axisLine={{
                      stroke: 'rgba(255,255,255,0.2)',
                    }}
                    tickLine={false}
                    minTickGap={35}
                    tickMargin={12}
                  />

                  <YAxis
                    domain={['auto', 'auto']}
                    tick={{
                      fill: '#aeb4c5',
                      fontSize: 10,
                    }}
                    axisLine={false}
                    tickLine={false}
                    tickMargin={10}
                    width={55}
                    tickFormatter={(value) =>
                      Number(value).toFixed(2)
                    }
                  />

                  <Tooltip
                    contentStyle={{
                      background: '#17151f',
                      border: '1px solid rgba(167,120,255,0.3)',
                      borderRadius: '10px',
                      color: '#eeeeF5',
                      fontSize: '12px',
                    }}
                    labelStyle={{
                      color: '#c4a5ff',
                      marginBottom: '5px',
                    }}
                    labelFormatter={(value) =>
                      formatDate(String(value))
                    }
                    formatter={(value) => [
                      Number(value).toFixed(2),
                      'NAV',
                    ]}
                  />

                  <Line
                    type="monotone"
                    dataKey="nav"
                    name="NAV"
                    stroke="#a78bfa"
                    strokeWidth={2.5}
                    strokeOpacity={1}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    dot={false}
                    activeDot={{
                      r: 5,
                      fill: '#c4a5ff',
                      stroke: '#17151f',
                      strokeWidth: 2,
                    }}
                    isAnimationActive={false}
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {metrics.observations > chartData.length && (
              <p className="chart-sampling-note">
                Chart display simplified to {chartData.length.toLocaleString('en-IN')} points
                from {metrics.observations.toLocaleString('en-IN')} observations.
                All performance metrics use the full dataset.
              </p>
            )}
          </>
        ) : (
          <div className="empty-chart">
            Historical NAV observations are not available for this fund.
          </div>
        )}
      </section>

      <section className="historical-returns-section">
        <div className="dashboard-section-heading">
          <div>
            <span className="eyebrow">PERIODIC PERFORMANCE</span>
            <h4>Historical returns</h4>
            <p>
              Returns calculated using the available NAV observations.
            </p>
          </div>
        </div>

        <div className="returns-grid">
          <ReturnCard
            label="30 observations"
            value={metrics.return_30_observations_pct}
            description="Approximately one month of trading observations"
          />

          <ReturnCard
            label="90 observations"
            value={metrics.return_90_observations_pct}
            description="Approximately three months of trading observations"
          />

          <ReturnCard
            label="252 observations"
            value={metrics.return_252_observations_pct}
            description="Approximately one year of trading observations"
          />
        </div>
      </section>

      <section className="data-quality-section">
        <div className="dashboard-section-heading">
          <div>
            <span className="eyebrow">DATA QUALITY</span>
            <h4>Dataset overview</h4>
            <p>
              Coverage and cleaning information for this fund.
            </p>
          </div>
        </div>

        <div className="quality-grid">
          <div className="quality-item">
            <span>Observations</span>
            <strong>{metrics.observations}</strong>
          </div>

          <div className="quality-item">
            <span>Date range</span>
            <strong className="quality-date">
              {formatDate(metrics.start_date)} – {formatDate(metrics.end_date)}
            </strong>
          </div>

          <div className="quality-item">
            <span>Original rows</span>
            <strong>{cleaning?.original_rows ?? '—'}</strong>
          </div>

          <div className="quality-item">
            <span>Final rows</span>
            <strong>{cleaning?.final_rows ?? '—'}</strong>
          </div>

          <div className="quality-item">
            <span>Removed rows</span>
            <strong>{cleaning?.removed_rows ?? '—'}</strong>
          </div>

          <div className="quality-item">
            <span>Invalid dates</span>
            <strong>{cleaning?.invalid_date_rows ?? '—'}</strong>
          </div>

          <div className="quality-item">
            <span>Invalid NAV values</span>
            <strong>{cleaning?.invalid_nav_rows ?? '—'}</strong>
          </div>

          <div className="quality-item">
            <span>Non-positive NAV</span>
            <strong>{cleaning?.non_positive_nav_rows ?? '—'}</strong>
          </div>

          <div className="quality-item">
            <span>Duplicate dates</span>
            <strong>{cleaning?.duplicate_date_rows ?? '—'}</strong>
          </div>
        </div>

        {warnings.length > 0 && (
          <div className="warnings-panel">
            <div className="warnings-heading">
              <AlertTriangle size={16} />
              <strong>Backend warnings</strong>
            </div>

            <ul>
              {warnings.map((warning, index) => (
                <li key={`${index}-${warning}`}>
                  {warning}
                </li>
              ))}
            </ul>
          </div>
        )}

        {warnings.length === 0 && (
          <div className="no-warnings">
            <ShieldCheck size={15} />
            No backend warnings reported for this dataset.
          </div>
        )}
      </section>
    </article>
  )
})

/* =========================================
   PREMIUM NAV ASSISTANT
========================================= */

function AssistantChat({
  messages,
  input,
  loading,
  error,
  selectedFundName,
  fundCount,
  onInputChange,
  onSend,
}: {
  messages: AssistantMessage[]
  input: string
  loading: boolean
  error: string
  selectedFundName: string
  fundCount: number
  onInputChange: (value: string) => void
  onSend: (event: FormEvent<HTMLFormElement>) => void
}) {
  const suggestions = [
    {
      title: 'Fund performance',
      description: 'Explain returns and risk',
      prompt: 'Explain this fund’s performance and risk.',
      icon: <TrendingUp size={17} />,
    },
    {
      title: 'Compare funds',
      description: 'Find the strongest performer',
      prompt: 'Compare the returns of all analyzed funds.',
      icon: <ChartNoAxesCombined size={17} />,
    },
    {
      title: 'Understand risk',
      description: 'Explain maximum drawdown',
      prompt: 'What does maximum drawdown mean?',
      icon: <Activity size={17} />,
    },
    {
      title: 'Lowest volatility',
      description: 'Compare fund stability',
      prompt: 'Which fund has the lowest volatility?',
      icon: <ShieldCheck size={17} />,
    },
  ]

  return (
    <section className="assistant-section">
      <div className="assistant-header">
        <div className="assistant-title">
          <div className="assistant-icon">
            <Sparkles size={21} />
          </div>

          <div className="assistant-heading-copy">
            <div className="assistant-heading-line">
              <h3>NAV Assistant</h3>
              <span className="assistant-ai-badge">AI POWERED</span>
            </div>

            <p>Your personal fund analytics companion</p>
          </div>
        </div>

        <div className="assistant-status">
          <span className="assistant-status-dot" />
          Ready to help
        </div>
      </div>

      <div className="assistant-context">
        <div className="assistant-context-item">
          <span className="assistant-context-icon">
            <Activity size={15} />
          </span>

          <div>
            <span className="assistant-context-label">
              CURRENT FUND
            </span>

            <strong title={selectedFundName}>
              {selectedFundName || 'No fund selected'}
            </strong>
          </div>
        </div>

        <div className="assistant-context-divider" />

        <div className="assistant-context-item">
          <span className="assistant-context-icon">
            <Layers3 size={15} />
          </span>

          <div>
            <span className="assistant-context-label">
              ANALYSIS SCOPE
            </span>

            <strong>
              {fundCount} {fundCount === 1 ? 'fund' : 'funds'} available
            </strong>
          </div>
        </div>
      </div>

      <div className="assistant-messages" aria-live="polite">
        {messages.length === 0 ? (
          <div className="assistant-welcome">
            <div className="assistant-welcome-icon">
              <Sparkles size={25} />
            </div>

            <span className="assistant-welcome-eyebrow">
              YOUR ANALYTICS COPILOT
            </span>

            <h4>What would you like to explore?</h4>

            <p>
              Ask me about performance, compare your funds, or
              understand the numbers behind your analysis.
            </p>

            <div className="assistant-suggestions">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion.title}
                  type="button"
                  className="assistant-suggestion"
                  onClick={() => onInputChange(suggestion.prompt)}
                  disabled={loading}
                >
                  <span className="assistant-suggestion-icon">
                    {suggestion.icon}
                  </span>

                  <span className="assistant-suggestion-copy">
                    <strong>{suggestion.title}</strong>
                    <small>{suggestion.description}</small>
                  </span>

                  <ArrowUpRight
                    className="assistant-suggestion-arrow"
                    size={16}
                  />
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="assistant-message-list">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`assistant-message assistant-message-${message.role}`}
              >
                <div className="assistant-message-avatar">
                  {message.role === 'assistant' ? (
                    <Sparkles size={16} />
                  ) : (
                    <MessageCircle size={16} />
                  )}
                </div>

                <div className="assistant-message-content">
                  <div className="assistant-message-meta">
                    <span>
                      {message.role === 'assistant'
                        ? 'NAV Assistant'
                        : 'You'}
                    </span>

                    {message.role === 'assistant' && (
                      <span className="assistant-message-tag">
                        AI
                      </span>
                    )}
                  </div>

                  {message.role === 'assistant' ? (
                    <AssistantMarkdown content={message.content} />
                  ) : (
                    <div className="assistant-user-message">
                      {message.content}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="assistant-message assistant-message-assistant">
                <div className="assistant-message-avatar">
                  <Sparkles size={16} />
                </div>

                <div className="assistant-message-content">
                  <div className="assistant-message-meta">
                    <span>NAV Assistant</span>
                  </div>

                  <p className="assistant-thinking">
                    <LoaderCircle className="spin" size={15} />
                    Reviewing your fund analysis...
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="assistant-error" role="alert">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      <form className="assistant-input-area" onSubmit={onSend}>
        <div className="assistant-composer">
          <textarea
            value={input}
            onChange={(event) => onInputChange(event.target.value)}
            placeholder="Ask anything about your fund analysis..."
            rows={2}
            maxLength={4000}
            disabled={loading}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }}
          />

          <button
            type="submit"
            disabled={loading || !input.trim()}
            aria-label="Send message"
            title="Send message"
          >
            {loading ? (
              <LoaderCircle className="spin" size={19} />
            ) : (
              <Send size={19} />
            )}
          </button>
        </div>

        <div className="assistant-composer-footer">
          <span>
            <kbd>Enter</kbd> to send · <kbd>Shift + Enter</kbd> for a new line
          </span>

          <span>{input.length}/4000</span>
        </div>
      </form>

      <p className="assistant-disclaimer">
        <ShieldCheck size={13} />
        AI-generated insights are informational, not investment advice.
      </p>
    </section>
  )
}

/* =========================================
   MAIN APPLICATION
========================================= */

function App() {
  const [files, setFiles] = useState<File[]>([])
  const [results, setResults] = useState<ApiResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedFundIndex, setSelectedFundIndex] = useState(0)

  const [page, setPage] = useState<'upload' | 'analysis' | 'assistant'>(() => {
    if (window.location.hash === '#/analysis') return 'analysis'
    if (window.location.hash === '#/assistant') return 'assistant'
    return 'upload'
  })

  const [storageReady, setStorageReady] = useState(false)
  const [storageError, setStorageError] = useState('')

  // Assistant state.
  const [assistantMessages, setAssistantMessages] = useState<AssistantMessage[]>(() =>
    loadAssistantMessages(),
  )
  const [assistantInput, setAssistantInput] = useState('')
  const [assistantLoading, setAssistantLoading] = useState(false)
  const [assistantError, setAssistantError] = useState('')

  useEffect(() => {
    let active = true

    const restoreAnalysis = async () => {
      try {
        const saved = await loadAnalysis<ApiResponse>()

        if (!active) return

        if (saved?.results) {
          setResults(saved.results)

          const restoredFundCount = Array.isArray(saved.results.funds)
            ? saved.results.funds.length
            : 0

          setSelectedFundIndex(
            restoredFundCount > 0
              ? Math.min(
                  Math.max(saved.selectedFundIndex, 0),
                  restoredFundCount - 1,
                )
              : 0,
          )
        } else if (
          window.location.hash === '#/analysis' ||
          window.location.hash === '#/assistant'
        ) {
          window.history.replaceState(
            null,
            '',
            `${window.location.pathname}${window.location.search}#/upload`,
          )

          setPage('upload')
        }
      } catch (err) {
        if (active) {
          setStorageError(
            err instanceof Error
              ? err.message
              : 'Could not restore the saved analysis.',
          )
        }
      } finally {
        if (active) setStorageReady(true)
      }
    }

    void restoreAnalysis()

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const syncPageFromUrl = () => {
      const hash = window.location.hash
      setPage(
        hash === '#/analysis'
          ? 'analysis'
          : hash === '#/assistant'
            ? 'assistant'
            : 'upload',
      )
    }

    window.addEventListener('popstate', syncPageFromUrl)
    window.addEventListener('hashchange', syncPageFromUrl)

    return () => {
      window.removeEventListener('popstate', syncPageFromUrl)
      window.removeEventListener('hashchange', syncPageFromUrl)
    }
  }, [])

  const navigate = (nextPage: 'upload' | 'analysis' | 'assistant') => {
    const nextHash =
      nextPage === 'analysis'
        ? '#/analysis'
        : nextPage === 'assistant'
          ? '#/assistant'
          : '#/upload'

    if (window.location.hash !== nextHash) {
      window.history.pushState(
        null,
        '',
        `${window.location.pathname}${window.location.search}${nextHash}`,
      )
    }

    setPage(nextPage)
  }

  useEffect(() => {
    try {
      window.localStorage.setItem(
        ASSISTANT_CHAT_STORAGE_KEY,
        JSON.stringify(assistantMessages),
      )
    } catch (err) {
      console.warn('Could not save assistant chat history.', err)
    }
  }, [assistantMessages])

  useEffect(() => {
    if (!storageReady || !results) return

    void saveAnalysis(results, selectedFundIndex).catch((err: unknown) => {
      setStorageError(
        err instanceof Error
          ? err.message
          : 'Could not save the selected fund.',
      )
    })
  }, [results, selectedFundIndex, storageReady])

  const funds = Array.isArray(results?.funds)
    ? results.funds
    : []

  const preparedFunds = useMemo<PreparedFund[]>(() => {
    return funds.map((fund) => ({
      ...fund,
      chartData: prepareChartData(fund.historical ?? []),
    }))
  }, [funds])

  const selectedFund =
    preparedFunds[selectedFundIndex] ?? preparedFunds[0]

  const invalidateAnalysis = () => {
    setResults(null)
    setSelectedFundIndex(0)
    setStorageError('')
    setAssistantMessages([])
    setAssistantInput('')
    setAssistantError('')

    void clearAnalysis().catch((err: unknown) => {
      setStorageError(
        err instanceof Error
          ? err.message
          : 'Could not clear the saved analysis.',
      )
    })

    navigate('upload')
  }

  const addFiles = (selectedFiles: FileList | null) => {
    if (!selectedFiles) return

    const supportedFiles = Array.from(selectedFiles).filter(
      (file) => /\.(csv|xls|xlsx)$/i.test(file.name),
    )

    if (supportedFiles.length > 0) {
      invalidateAnalysis()
    }

    setFiles((current) => {
      const combined = [...current, ...supportedFiles]

      return combined.filter(
        (file, index) =>
          combined.findIndex(
            (item) =>
              item.name === file.name &&
              item.size === file.size,
          ) === index,
      )
    })

    setError(
      supportedFiles.length === selectedFiles.length
        ? ''
        : 'Some files were skipped. Only CSV, XLS and XLSX files are supported.',
    )
  }

  const removeFile = (fileName: string) => {
    setFiles((current) =>
      current.filter((file) => file.name !== fileName),
    )

    invalidateAnalysis()
    setError('')
  }

  const clearWorkspace = () => {
    setFiles([])
    setResults(null)
    setSelectedFundIndex(0)
    setError('')
    setStorageError('')
    setAssistantMessages([])
    setAssistantInput('')
    setAssistantError('')

    navigate('upload')

    void clearAnalysis().catch((err: unknown) => {
      setStorageError(
        err instanceof Error
          ? err.message
          : 'Could not clear the saved analysis.',
      )
    })
  }

  const analyzeFiles = async () => {
    if (files.length === 0) {
      setError('Please select at least one supported fund file.')
      return
    }

    setLoading(true)
    setError('')
    setResults(null)
    setSelectedFundIndex(0)
    setAssistantMessages([])
    setAssistantInput('')
    setAssistantError('')

    try {
      await clearAnalysis()
      setStorageError('')
    } catch (storageErr) {
      setStorageError(
        storageErr instanceof Error
          ? storageErr.message
          : 'Could not clear the previous saved analysis.',
      )
    }

    try {
      const formData = new FormData()

      files.forEach((file) => {
        formData.append('files', file)
      })

      const response = await fetch(
        `${API_BASE_URL}/api/v1/analyze`,
        {
          method: 'POST',
          body: formData,
        },
      )

      const data = await response.json()

      if (!response.ok) {
        const detail = data.detail

        throw new Error(
          typeof detail === 'string'
            ? detail
            : detail?.message || 'The analysis request failed.',
        )
      }

      const apiResults = data as ApiResponse

      try {
        await saveAnalysis(apiResults, 0)
        setStorageError('')
      } catch (storageErr) {
        setStorageError(
          storageErr instanceof Error
            ? storageErr.message
            : 'Analysis completed, but its results could not be saved.',
        )
      }

      setResults(apiResults)
      setSelectedFundIndex(0)
      navigate('analysis')
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Something went wrong while analyzing your files.',
      )
    } finally {
      setLoading(false)
    }
  }

  // Send a question with selected-fund and cross-fund context.
  const sendAssistantMessage = async (
    event: FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault()

    const question = assistantInput.trim()

    if (!question || assistantLoading || !results) return

    const userMessage: AssistantMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      content: question,
    }

    const conversation = [...assistantMessages, userMessage]

    setAssistantMessages(conversation)
    setAssistantInput('')
    setAssistantError('')
    setAssistantLoading(true)

    const fundContext = funds.map((fund) => ({
      fund_name: fund.fund_name,
      source_file: fund.source_file,
      metrics: fund.metrics,
      cleaning: fund.cleaning,
      warnings: fund.warnings,
    }))

    const selectedFundContext = selectedFund
      ? {
          fund_name: selectedFund.fund_name,
          source_file: selectedFund.source_file,
          metrics: selectedFund.metrics,
          cleaning: selectedFund.cleaning,
          warnings: selectedFund.warnings,
        }
      : null

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/assistant/chat`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: question,

            history: assistantMessages.map((message) => ({
              role: message.role,
              content: message.content,
            })),

            context: {
              selected_fund: selectedFundContext,
              funds: fundContext,
              fund_count: funds.length,
              product_name: results.product_name ?? null,
            },
          }),
        },
      )

      const data = await response.json() as AssistantApiResponse

      if (!response.ok) {
        const detail = data.detail

        throw new Error(
          typeof detail === 'string'
            ? detail
            : detail?.message || 'The assistant could not process your question.',
        )
      }

      const reply =
        data.reply ||
        data.answer ||
        data.response ||
        data.message

      if (!reply) {
        throw new Error(
          'The assistant returned an empty response. Please try again.',
        )
      }

      setAssistantMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-assistant`,
          role: 'assistant',
          content: reply,
        },
      ])
    } catch (err) {
      setAssistantError(
        err instanceof Error
          ? err.message
          : 'Could not connect to the analytics assistant.',
      )
    } finally {
      setAssistantLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <ChartNoAxesCombined size={23} />
          </div>

          <div>
            <h1>Fund Board</h1>
            <span>Fund intelligence workspace</span>
          </div>
        </div>

        <div className="sidebar-section">
          <p className="section-label">WORKSPACE</p>

          <div
            className={`nav-item ${page === 'upload' ? 'active' : ''}`}
            role="button"
            tabIndex={0}
            onClick={() => navigate('upload')}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                navigate('upload')
              }
            }}
          >
            <CloudUpload size={18} />
            <span>Upload funds</span>
          </div>

          <div
            className={`nav-item ${page === 'analysis' ? 'active' : ''}`}
            role="button"
            tabIndex={0}
            aria-disabled={!results}
            onClick={() => {
              if (results) navigate('analysis')
            }}
            onKeyDown={(event) => {
              if (results && (event.key === 'Enter' || event.key === ' ')) {
                event.preventDefault()
                navigate('analysis')
              }
            }}
          >
            <Layers3 size={18} />
            <span>Fund analysis</span>
          </div>

          <div
            className={`nav-item ${page === 'assistant' ? 'active' : ''}`}
            role="button"
            tabIndex={results ? 0 : -1}
            aria-disabled={!results}
            onClick={() => {
              if (results) navigate('assistant')
            }}
            onKeyDown={(event) => {
              if (results && (event.key === 'Enter' || event.key === ' ')) {
                event.preventDefault()
                navigate('assistant')
              }
            }}
          >
            <MessageCircle size={18} />
            <span>AI Assistant</span>
          </div>
        </div>

        {funds.length > 0 && (
          <div className="sidebar-funds">
            <div className="sidebar-funds-heading">
              <p className="section-label">ANALYZED FUNDS</p>
              <span>{funds.length}</span>
            </div>

            <div className="fund-nav-list">
              {funds.map((fund, index) => (
                <button
                  key={`${fund.source_file || fund.fund_name}-${index}`}
                  type="button"
                  className={`fund-nav-item ${
                    selectedFundIndex === index ? 'selected' : ''
                  }`}
                  onClick={() => {
                    setSelectedFundIndex(index)
                    navigate('analysis')
                  }}
                  title={fund.fund_name}
                >
                  <span className="fund-nav-number">
                    {String(index + 1).padStart(2, '0')}
                  </span>

                  <span className="fund-nav-name">
                    {fund.fund_name || `Fund ${index + 1}`}
                  </span>

                  <ChevronRight
                    className="fund-nav-chevron"
                    size={15}
                  />
                </button>
              ))}
            </div>
          </div>
        )}
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="breadcrumb">
              WORKSPACE / {page === 'analysis' ? 'ANALYSIS' : page === 'assistant' ? 'AI ASSISTANT' : 'UPLOAD'}
            </span>

            <h2>
              {page === 'analysis' ? 'Analysis results' : page === 'assistant' ? 'AI Assistant' : 'Fund analysis'}
            </h2>
          </div>

          {page !== 'upload' && (
            <button
              className="text-button"
              type="button"
              onClick={() => navigate('upload')}
            >
              Back to upload
            </button>
          )}
        </header>

        {page === 'upload' && (
          <>
            <section className="welcome-section">
              <div>
                <div className="eyebrow">
                  <Sparkles size={14} />
                  NAV ANALYTICS
                </div>

                <h1>
                  Understand your fund.
                  <br />
                  Explore its history.
                </h1>

                <p>
                  Analyze historical mutual fund NAV data, understand
                  performance, and examine risk across the available
                  observations.
                </p>
              </div>

              <div className="welcome-art">
                <div className="art-circle">
                  <ChartNoAxesCombined
                    size={62}
                    strokeWidth={1.1}
                  />
                </div>
              </div>
            </section>

            <section className="upload-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">GET STARTED</span>
                  <h3>Upload your fund data</h3>
                  <p>
                    Upload one file per fund. Each fund is analyzed independently.
                  </p>
                </div>

                <div className="csv-label">
                  <FileSpreadsheet size={16} />
                  Excel &amp; CSV
                </div>
              </div>

              <label
                className="upload-zone"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault()
                  addFiles(event.dataTransfer.files)
                }}
              >
                <input
                  type="file"
                  accept=".csv,.xls,.xlsx,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                  multiple
                  onChange={(event) => {
                    addFiles(event.target.files)
                    event.target.value = ''
                  }}
                />

                <div className="upload-icon">
                  <CloudUpload size={28} />
                </div>

                <strong>Drop your fund files here</strong>
                <span>or click to browse from your computer</span>

                <small>
                  Supported formats: CSV, XLS and XLSX · Required columns:
                  Date and NAV
                </small>
              </label>

              {files.length > 0 && (
                <div className="file-list">
                  <div className="file-list-heading">
                    <strong>Selected files ({files.length})</strong>

                    <button
                      className="text-button"
                      onClick={clearWorkspace}
                      type="button"
                    >
                      Clear all
                    </button>
                  </div>

                  {files.map((file) => (
                    <div
                      className="file-row"
                      key={`${file.name}-${file.size}`}
                    >
                      <FileSpreadsheet size={18} />

                      <span>{file.name}</span>

                      <small>
                        {(file.size / 1024).toFixed(1)} KB
                      </small>

                      <button
                        className="icon-button"
                        onClick={() => removeFile(file.name)}
                        type="button"
                        aria-label={`Remove ${file.name}`}
                      >
                        <X size={16} />
                      </button>
                    </div>
                  ))}

                  <button
                    className="analyze-button"
                    onClick={analyzeFiles}
                    disabled={loading}
                    type="button"
                  >
                    {loading ? (
                      <>
                        <LoaderCircle className="spin" size={18} />
                        Analyzing funds...
                      </>
                    ) : (
                      <>
                        Analyze {files.length}{' '}
                        {files.length === 1 ? 'fund' : 'funds'}
                        <ArrowUpRight size={18} />
                      </>
                    )}
                  </button>
                </div>
              )}

              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}

              <button
                className="clear-workspace-button"
                onClick={clearWorkspace}
                disabled={loading}
                type="button"
              >
                <Trash2 size={16} />
                Clear workspace
              </button>
            </section>
          </>
        )}

        {page === 'analysis' && (
          results ? (
            <>
              <section className="results-section">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">ANALYSIS RESULTS</span>
                    <h3>Fund performance overview</h3>
                    <p>
                      {funds.length}{' '}
                      {funds.length === 1 ? 'fund' : 'funds'} analyzed
                      independently using the uploaded NAV data.
                    </p>
                  </div>

                  <div className="success-label">
                    <ShieldCheck size={16} />
                    Analysis completed
                  </div>
                </div>

                {funds.length > 0 && selectedFund ? (
                  <div className="selected-fund-workspace">
                    {results.product_name && (
                      <div className="product-heading">
                        <span className="eyebrow">PRODUCT</span>
                        <h3>{results.product_name}</h3>
                      </div>
                    )}

                    <div className="selected-fund-heading">
                      <div>
                        <span className="eyebrow">
                          SELECTED FUND ·{' '}
                          {String(selectedFundIndex + 1).padStart(2, '0')}
                        </span>
                      </div>

                      <span className="selected-fund-count">
                        {selectedFundIndex + 1} of {funds.length}
                      </span>
                    </div>

                    <div className="fund-dashboard-stack">
                      {preparedFunds.map((fund, index) => {
                        const isSelected = index === selectedFundIndex

                        return (
                          <div
                            key={`${fund.source_file || fund.fund_name}-${index}`}
                            className={`fund-dashboard-panel ${
                              isSelected ? 'is-active' : 'is-inactive'
                            }`}
                            aria-hidden={!isSelected}
                          >
                            <FundDashboard fund={fund} />
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="empty-results">
                    The backend returned successfully, but no fund analysis
                    records were found in the response.
                  </div>
                )}

                {results.file_errors &&
                  results.file_errors.length > 0 && (
                    <div className="file-errors-panel">
                      <strong>
                        Files that could not be analyzed
                      </strong>

                      {results.file_errors.map((item, index) => (
                        <p key={`${item.filename}-${index}`}>
                          <strong>
                            {item.filename || 'Unknown file'}:
                          </strong>{' '}
                          {item.error || 'An error occurred while processing this file.'}
                        </p>
                      ))}
                    </div>
                  )}
              </section>

            </>
          ) : (
            <section className="page-state-panel">
              {storageReady ? (
                <>
                  <h3>No saved analysis found</h3>
                  <p>
                    Upload your fund data and run an analysis to view results here.
                  </p>

                  <button
                    className="text-button"
                    type="button"
                    onClick={() => navigate('upload')}
                  >
                    Go to upload
                  </button>
                </>
              ) : (
                <>
                  <h3>Restoring your analysis...</h3>
                  <p>
                    Please wait while your saved fund results are loaded.
                  </p>
                </>
              )}
            </section>
          )
        )}

        {page === 'assistant' && (
          results && funds.length > 0 ? (
            <div className="assistant-page">
              <div className="assistant-page-intro">
                <span className="eyebrow"><Sparkles size={14} /> FUND INTELLIGENCE</span>
                <h3>Ask your NAV Assistant</h3>
                <p>
                  Explore the selected fund, compare every analyzed fund, and
                  get clear explanations of performance and risk metrics.
                </p>
              </div>

              <AssistantChat
                messages={assistantMessages}
                input={assistantInput}
                loading={assistantLoading}
                error={assistantError}
                selectedFundName={selectedFund?.fund_name ?? ''}
                fundCount={funds.length}
                onInputChange={setAssistantInput}
                onSend={sendAssistantMessage}
              />
            </div>
          ) : (
            <section className="page-state-panel">
              <h3>Analyze funds to use the AI Assistant</h3>
              <p>
                Upload and analyze at least one fund file. Your assistant will
                then be able to explain its metrics and compare funds.
              </p>
              <button
                className="text-button"
                type="button"
                onClick={() => navigate('upload')}
              >
                Go to upload
              </button>
            </section>
          )
        )}

        {storageError && (
          <div className="error-message" role="status">
            Saved analysis storage: {storageError}
          </div>
        )}

        <footer className="disclaimer">
          <ShieldCheck size={16} />

          <span>
            Historical performance does not guarantee future results.
            This dashboard is for analysis and informational purposes
            only, not investment advice.
          </span>
        </footer>
      </main>
    </div>
  )
}

export default App