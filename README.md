# Amazon Review Data Control & Analysis System

A comprehensive system for Amazon review data modeling, deterministic score generation using PyMC, and advanced analysis with stylistic matplotlib visualizations.

## 🚀 Features

### Data Generation
- **Pydantic Models**: Robust `AmazonReview` model with comprehensive validation
- **PyMC-based Score Generation**: Deterministic rating and helpful vote generation using probabilistic modeling
- **Realistic Data**: Generates placeholder data for all required Amazon review fields

### Analysis Modules
- **Temporal Analysis**: Review spikes detection, seasonality patterns, helpful vote trends, keyword frequency changes
- **User Behavior Analysis**: Superuser identification, review impact analysis, helpfulness patterns, user preferences
- **Product Analysis**: Ingredient correlation analysis, organic product trends, quality distribution indicators
- **Comprehensive Integration**: Combined analysis with insights generation and export capabilities

### Visualization System
- **Stylistic Matplotlib Plots**: Professional, publication-ready visualizations
- **Complete Coverage**: Plots for all analysis components
- **Summary Dashboard**: Integrated overview of all key metrics and findings
- **High-Quality Output**: 300 DPI plots saved to customizable output directories

## 📊 Generated Plots

The visualization system creates the following plot types:

### Temporal Analysis Plots
- **Review Spikes Analysis**: Monthly spike frequency and distribution
- **Seasonal Patterns**: Product mention patterns across seasons
- **Helpful Vote Trends**: Trending products with increasing helpfulness
- **Keyword Trends**: Frequency changes and concerning patterns

### User Behavior Plots
- **Superuser Analysis**: Comparison metrics and advantage multipliers
- **Helpfulness Characteristics**: Text length, rating distribution, verified purchase impact
- **User Preferences**: Top requested features and trend analysis
- **Review Impact**: Sales velocity correlation and impact distribution

### Product Analysis Plots
- **Ingredient Correlations**: Positive/negative correlations and effect sizes
- **Organic Trends**: Trend analysis, rating comparisons, keyword frequency
- **Quality Distribution**: Quality patterns, indicators correlation, comparisons

### Summary Dashboard
- **Key Metrics Overview**: Dataset size, spike days, superusers, organic percentage
- **Key Findings & Alerts**: Important discoveries and concerning patterns
- **Mini Visualizations**: Rating distribution, temporal trends, quality breakdown
- **Recommendations**: Actionable insights for business decisions

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd AmazonDataControl

# Install dependencies using PDM
pdm install

# Or install manually with pip
pip install pydantic pymc matplotlib seaborn pandas numpy scipy
```

## 📖 Usage Examples

### Quick Visualization Demo
```bash
pdm run python visualization_demo.py
```

### Comprehensive Analysis
```bash
pdm run python comprehensive_analysis_example.py
```

### Basic Data Generation
```bash
pdm run python example.py
```

## 📁 Project Structure

```
AmazonDataControl/
├── src/adc/
│   ├── models.py                    # Pydantic models
│   ├── generation/
│   │   └── score_generator.py       # PyMC-based data generation
│   └── analysis/
│       ├── temporal_analyzer.py     # Temporal pattern analysis
│       ├── user_behavior_analyzer.py # User behavior analysis
│       ├── product_analyzer.py      # Product analysis
│       ├── comprehensive_analyzer.py # Integrated analysis
│       └── visualizer.py           # Visualization system
├── example.py                       # Basic generation example
├── comprehensive_analysis_example.py # Full analysis demo
├── visualization_demo.py            # Visualization-focused demo
└── analysis_plots/                  # Generated visualization output
```

## 🎨 Visualization Features

- **Professional Styling**: Uses seaborn styling with custom color palettes
- **High Resolution**: All plots saved at 300 DPI for publication quality
- **Responsive Design**: Automatic layout adjustment and proper spacing
- **Comprehensive Coverage**: Every analysis component has corresponding visualizations
- **Summary Integration**: Unified dashboard showing all key metrics
- **File Export**: All plots automatically saved to organized output directory

## 📈 Analysis Capabilities

### Temporal Analysis
- Positive review spike detection with threshold analysis
- Seasonal product trend identification
- Helpful vote increase tracking
- Keyword frequency trend analysis with alerts

### User Behavior Analysis
- Superuser identification and influence measurement
- Review helpfulness pattern analysis
- User product preference extraction
- Review impact on sales velocity correlation

### Product Analysis
- Ingredient-rating correlation analysis
- Organic/natural product trend tracking
- Product quality indicator development
- Feature preference analysis

## 🔧 Customization

The system is designed for easy extension:
- Add new analysis modules by implementing the base analyzer pattern
- Extend visualization by adding new plot methods to `AnalysisVisualizer`
- Customize data generation by modifying PyMC model parameters
- Configure output formats and styling through matplotlib settings

## 📊 Output Files

Running the analysis generates:
- **JSON Results**: Complete analysis results with metadata
- **Summary Report**: Human-readable text summary
- **Visualization Plots**: Professional matplotlib figures
- **Analysis Logs**: Detailed processing information

## 🎯 Key Use Cases

- **E-commerce Analytics**: Understanding review patterns and customer behavior
- **Product Quality Assessment**: Identifying quality indicators and correlations
- **Marketing Intelligence**: Detecting trends and optimal timing
- **Customer Insights**: Analyzing user preferences and influence patterns
- **Quality Assurance**: Monitoring review authenticity and helpfulness

---

Built with PyMC, Pydantic, Matplotlib, and modern Python data science tools.
