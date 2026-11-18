import { memo, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
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
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
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
import { clearAnalysis, loadAnalysis, saveAnalysis } from './analysisStorage'
import './index.css'

// Use the configured Render API in production. When this variable is
// unset locally, Vite's /api proxy continues to handle development requests.
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

// Keep the chart reasonably light without changing the underlying analysis.
// First and last observations are always retained. Each bucket also retains
// its lowest and highest NAV values to preserve visible peaks and troughs.

const MAX_CHART_POINTS = 1000

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

    // Preserve the chronological order of the selected extremes.
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

// Memoized dashboard. Each fund stays mounted after analysis,
// so switching tabs does not repeatedly recreate its chart.

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
          value={formatNumber(metrics.latest_nav, 4)}
          description={`As of ${formatDate(metrics.end_date)}`}
          icon={<Activity size={17} />}
        />

        <MetricCard
          label="Starting NAV"
          value={formatNumber(metrics.starting_nav, 4)}
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
                      Number(value).toFixed(4),
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

function App() {
  const [files, setFiles] = useState<File[]>([])
  const [results, setResults] = useState<ApiResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedFundIndex, setSelectedFundIndex] = useState(0)
  const [currentPath, setCurrentPath] = useState(() =>
    window.location.pathname === '/analysis' ? '/analysis' : '/',
  )
  const [isRestoring, setIsRestoring] = useState(true)

  // Restore the last successful analysis when the app is opened or refreshed.
  useEffect(() => {
    let cancelled = false

    const restoreAnalysis = async () => {
      try {
        const saved = await loadAnalysis<ApiResponse>()

        if (!cancelled && saved?.results) {
          const savedFunds = Array.isArray(saved.results.funds)
            ? saved.results.funds
            : []
          const safeIndex = Math.min(
            Math.max(saved.selectedFundIndex, 0),
            Math.max(savedFunds.length - 1, 0),
          )

          setResults(saved.results)
          setSelectedFundIndex(safeIndex)
        }
      } catch {
        if (!cancelled) {
          setError('Your saved analysis could not be restored in this browser.')
        }
      } finally {
        if (!cancelled) setIsRestoring(false)
      }
    }

    const handlePopState = () => {
      setCurrentPath(
        window.location.pathname === '/analysis' ? '/analysis' : '/',
      )
    }

    window.addEventListener('popstate', handlePopState)
    void restoreAnalysis()

    return () => {
      cancelled = true
      window.removeEventListener('popstate', handlePopState)
    }
  }, [])

  const navigateTo = (path: '/' | '/analysis') => {
    if (window.location.pathname !== path) {
      window.history.pushState({}, '', path)
    }
    setCurrentPath(path)
  }

  const selectFund = (index: number) => {
    setSelectedFundIndex(index)
    if (results) {
      void saveAnalysis(results, index).catch(() => {
        setError('The selected fund could not be saved for your next visit.')
      })
    }
  }

  const funds = Array.isArray(results?.funds)
    ? results.funds
    : []

  // Prepare and downsample each fund's chart data once per analysis.
  // These references remain stable while switching between funds.

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
    void clearAnalysis().catch(() => undefined)
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
    void clearAnalysis().catch(() => undefined)
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
    void clearAnalysis().catch(() => undefined)

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

      setResults(apiResults)
      setSelectedFundIndex(0)

      try {
        await saveAnalysis(apiResults, 0)
      } catch {
        setError(
          'Analysis completed, but this browser could not save a persistent copy. Keep this tab open to view the results.',
        )
      }

      navigateTo('/analysis')
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

          <button
            className={`nav-item ${currentPath === '/' ? 'active' : ''}`}
            type="button"
            onClick={() => navigateTo('/')}
          >
            <CloudUpload size={18} />
            <span>Upload data</span>
          </button>

          <button
            className={`nav-item ${currentPath === '/analysis' ? 'active' : ''}`}
            type="button"
            onClick={() => navigateTo('/analysis')}
          >
            <Layers3 size={18} />
            <span>Fund analysis</span>
          </button>
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
                  onClick={() => selectFund(index)}
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
              WORKSPACE / {currentPath === '/analysis' ? 'ANALYSIS' : 'UPLOAD'}
            </span>

            <h2>{currentPath === '/analysis' ? 'Fund analysis' : 'Upload fund data'}</h2>
          </div>

        </header>

        {currentPath === '/' && (
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
        </section>
          </>
        )}

        {currentPath === '/analysis' && (
          isRestoring ? (
            <div className="page-state-panel">
              <LoaderCircle className="spin" size={22} />
              <strong>Restoring your saved analysis...</strong>
              <span>Your previous results are being loaded from this browser.</span>
            </div>
          ) : results ? (
          <>
          {error && <div className="error-message" role="alert">{error}</div>}
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

                {/* All dashboards stay mounted. Only visibility changes. */}

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
            <div className="page-state-panel">
              <Layers3 size={24} />
              <strong>No saved analysis found</strong>
              <span>Upload your fund files and run an analysis to see results here.</span>
              <button type="button" onClick={() => navigateTo('/')}>
                Go to file upload
                <ArrowUpRight size={16} />
              </button>
            </div>
          )
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