import { useState } from 'react'
import {
  Activity,
  ArrowUpRight,
  ChartNoAxesCombined,
  CloudUpload,
  FileSpreadsheet,
  Layers3,
  LoaderCircle,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import './index.css'

type ApiResponse = Record<string, unknown>

function App() {
  const [files, setFiles] = useState<File[]>([])
  const [results, setResults] = useState<ApiResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Accept CSV, XLS and XLSX files.
  const addFiles = (selectedFiles: FileList | null) => {
    if (!selectedFiles) return

    const supportedFiles = Array.from(selectedFiles).filter((file) =>
      /\.(csv|xls|xlsx)$/i.test(file.name),
    )

    setFiles((current) => {
      const combined = [...current, ...supportedFiles]

      return combined.filter(
        (file, index) =>
          combined.findIndex((item) => item.name === file.name) === index,
      )
    })

    setError(
      supportedFiles.length === selectedFiles.length
        ? ''
        : 'Some files were skipped. Only CSV, XLS and XLSX files are supported.',
    )
  }

  // Remove an individual file.
  const removeFile = (fileName: string) => {
    setFiles((current) =>
      current.filter((file) => file.name !== fileName),
    )
  }

  // Upload selected files to the FastAPI backend.
  const analyzeFiles = async () => {
    if (files.length === 0) {
      setError('Please select at least one supported fund file.')
      return
    }

    setLoading(true)
    setError('')
    setResults(null)

    try {
      const formData = new FormData()

      files.forEach((file) => {
        formData.append('files', file)
      })

      const response = await fetch('/api/v1/analyze', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'The analysis request failed.',
        )
      }

      setResults(data as ApiResponse)
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

  // Determine the number of funds returned by the backend.
  const resultCount = Array.isArray(results?.funds)
    ? results.funds.length
    : Array.isArray(results?.results)
      ? results.results.length
      : 0

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <ChartNoAxesCombined size={23} />
          </div>

          <div>
            <h1>Operation-NAV</h1>
            <span>Fund intelligence workspace</span>
          </div>
        </div>

        <div className="sidebar-section">
          <p className="section-label">WORKSPACE</p>

          <div className="nav-item active">
            <Layers3 size={18} />
            <span>Fund analysis</span>
          </div>

          <div className="nav-item">
            <Activity size={18} />
            <span>Forecast insights</span>
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="status-indicator" />

          <div>
            <strong>Local environment</strong>
            <span>API workspace</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">
        {/* Header */}
        <header className="topbar">
          <div>
            <span className="breadcrumb">WORKSPACE / ANALYSIS</span>
            <h2>Fund analysis</h2>
          </div>

          <div className="topbar-status">
            <span className="live-dot" />
            Backend connected
          </div>
        </header>

        {/* Welcome section */}
        <section className="welcome-section">
          <div>
            <div className="eyebrow">
              <Sparkles size={14} />
              NAV ANALYTICS
            </div>

            <h1>
              Understand the past.
              <br />
              Explore what lies ahead.
            </h1>

            <p>
              Analyze historical mutual fund NAV data and explore
              independently generated 12-month forecasts.
            </p>
          </div>

          <div className="welcome-art">
            <div className="art-circle">
              <ChartNoAxesCombined size={62} strokeWidth={1.1} />
            </div>
          </div>
        </section>

        {/* File upload section */}
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

          {/* Drag and drop / file selection */}
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
              Supported formats: CSV, XLS and XLSX · Required columns: Date
              and NAV
            </small>
          </label>

          {/* Selected files */}
          {files.length > 0 && (
            <div className="file-list">
              <div className="file-list-heading">
                <strong>Selected files ({files.length})</strong>

                <button
                  className="text-button"
                  onClick={() => setFiles([])}
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

              {/* Analyze button */}
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

          {/* Error message */}
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </section>

        {/* Backend analysis response */}
        {results && (
          <section className="results-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">ANALYSIS RESPONSE</span>
                <h3>Analysis completed</h3>

                <p>
                  The backend returned a response successfully.
                  {resultCount > 0 &&
                    ` ${resultCount} fund records detected.`}
                </p>
              </div>

              <div className="success-label">
                <ShieldCheck size={16} />
                API success
              </div>
            </div>

            <div className="response-panel">
              <div className="response-header">
                <strong>Backend response</strong>
                <span>JSON</span>
              </div>

              <pre>
                {JSON.stringify(results, null, 2)}
              </pre>
            </div>
          </section>
        )}

        {/* Disclaimer */}
        <footer className="disclaimer">
          <ShieldCheck size={16} />

          <span>
            Forecasts are exploratory estimates, not guarantees or investment
            recommendations.
          </span>
        </footer>
      </main>
    </div>
  )
}

export default App