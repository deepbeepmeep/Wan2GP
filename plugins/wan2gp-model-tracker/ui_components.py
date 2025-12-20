"""
UI Components for Wan2GP Model Tracker

Reusable Gradio UI components and helper functions for building the interface.
"""

from typing import Dict, List, Tuple, Optional

# Lazy imports to avoid blocking during plugin discovery
try:
    from performance_db import PerformanceDatabase
except ImportError:
    from .performance_db import PerformanceDatabase


def create_model_table_data(analyzer, models_list: List[str],
                            status_filter: str = "all",
                            speed_filter: str = "all",
                            vram_filter: str = "all",
                            sort_by: str = "name"):
    """
    Create pandas DataFrame for model table display

    Args:
        analyzer: ModelAnalyzer instance
        models_list: List of model type identifiers
        status_filter: "all", "downloaded", "missing", "partial"
        speed_filter: "all", "fast", "medium", "slow"
        vram_filter: "all", "≤12GB", "≤24GB", ">24GB"
        sort_by: "name", "status", "speed", "vram", "quality"

    Returns:
        pandas DataFrame with columns: Status, Model Name, Speed, VRAM (GB), Quality, Size (GB), Best For
    """
    import pandas as pd

    rows = []

    for model_type in models_list:
        # Get model data
        status = analyzer.get_model_status(model_type)
        metrics = PerformanceDatabase.get_metrics(model_type)
        model_name = analyzer.get_model_name(model_type)
        size_gb = analyzer.get_model_size_gb(model_type)

        # Apply filters
        if status_filter != "all" and status != status_filter:
            continue

        if speed_filter != "all" and metrics["speed_tier"] != speed_filter:
            continue

        if vram_filter != "all":
            vram_rec = metrics["vram_recommended_gb"]
            if vram_filter == "≤12GB" and vram_rec > 12:
                continue
            elif vram_filter == "≤24GB" and vram_rec > 24:
                continue
            elif vram_filter == ">24GB" and vram_rec <= 24:
                continue

        # Format data for display
        status_icon = get_status_icon(status)
        speed_display = f"{metrics['speed_tier'].capitalize()}"
        vram_display = f"{metrics['vram_min_gb']}-{metrics['vram_recommended_gb']}"
        quality_display = metrics['quality_tier'].capitalize()
        size_display = f"{size_gb}" if status == "downloaded" else f"~{size_gb}"

        rows.append({
            "Status": status_icon,
            "Model Name": model_name,
            "Type": model_type,
            "Speed": speed_display,
            "VRAM (GB)": vram_display,
            "Quality": quality_display,
            "Size (GB)": size_display,
            "Best For": metrics["best_for"]
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    if df.empty:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["Status", "Model Name", "Type", "Speed", "VRAM (GB)",
                                     "Quality", "Size (GB)", "Best For"])

    # Apply sorting
    if sort_by == "name":
        df = df.sort_values("Model Name")
    elif sort_by == "status":
        # Downloaded first, then partial, then missing
        status_order = {"✓": 0, "◐": 1, "✗": 2, "?": 3}
        df["_sort_key"] = df["Status"].map(status_order)
        df = df.sort_values("_sort_key").drop("_sort_key", axis=1)
    elif sort_by == "speed":
        speed_order = {"Fast": 0, "Medium": 1, "Slow": 2}
        df["_sort_key"] = df["Speed"].map(speed_order)
        df = df.sort_values("_sort_key").drop("_sort_key", axis=1)
    elif sort_by == "vram":
        # Extract minimum VRAM for sorting
        df["_sort_key"] = df["VRAM (GB)"].apply(lambda x: int(x.split('-')[0]))
        df = df.sort_values("_sort_key").drop("_sort_key", axis=1)
    elif sort_by == "quality":
        quality_order = {"Low": 0, "Medium": 1, "High": 2, "Highest": 3}
        df["_sort_key"] = df["Quality"].map(quality_order)
        df = df.sort_values("_sort_key").drop("_sort_key", axis=1)

    return df


def get_status_icon(status: str) -> str:
    """Get icon for model status"""
    icons = {
        "downloaded": "✓",
        "partial": "◐",
        "missing": "✗",
        "unknown": "?"
    }
    return icons.get(status, "?")


def format_model_details(analyzer, model_type: str) -> str:
    """
    Format detailed model information as markdown

    Args:
        analyzer: ModelAnalyzer instance
        model_type: Model type identifier

    Returns:
        Formatted markdown string
    """
    if not model_type or model_type == "Select a model...":
        return "Select a model from the dropdown to view details."

    # Get model data
    status = analyzer.get_model_status(model_type)
    metrics = PerformanceDatabase.get_metrics(model_type)
    model_name = analyzer.get_model_name(model_type)
    description = analyzer.get_model_description(model_type)
    files = analyzer.get_model_files(model_type)
    urls = analyzer.get_download_urls(model_type)
    size_gb = analyzer.get_model_size_gb(model_type)

    # Build markdown
    md = f"# {model_name}\n\n"
    md += f"**Model Type:** `{model_type}`\n\n"
    md += f"**Status:** {get_status_icon(status)} {status.capitalize()}\n\n"

    if description:
        md += f"**Description:** {description}\n\n"

    md += "---\n\n"
    md += "## Performance Metrics\n\n"
    md += f"- **Speed Tier:** {metrics['speed_tier'].capitalize()} "
    md += f"({metrics['speed_multiplier']}x base speed)\n"
    md += f"- **VRAM Required:** {metrics['vram_min_gb']}GB minimum, "
    md += f"{metrics['vram_recommended_gb']}GB recommended\n"
    md += f"- **Quality Tier:** {metrics['quality_tier'].capitalize()}\n"
    md += f"- **Total Size:** {size_gb} GB\n"
    md += f"- **Best For:** {metrics['best_for']}\n"

    if metrics['notes']:
        md += f"\n*{metrics['notes']}*\n"

    # Files section
    md += "\n---\n\n"
    md += "## Files\n\n"
    if files:
        for file_info in files:
            icon = "✓" if file_info["status"] == "downloaded" else "✗"
            size_str = f"{file_info['size_gb']:.2f}GB" if file_info["size_gb"] else "unknown size"
            md += f"- {icon} `{file_info['filename']}` ({size_str})\n"
            if file_info["path"]:
                md += f"  - Path: `{file_info['path']}`\n"
            else:
                md += f"  - URL: {file_info['url']}\n"
    else:
        md += "*No file information available*\n"

    # Recommendations section
    md += "\n---\n\n"
    md += "## Recommendations\n\n"

    if metrics['similar_faster']:
        md += "**Faster alternatives:** "
        md += ", ".join(f"`{m}`" for m in metrics['similar_faster'])
        md += "\n\n"

    if metrics['similar_higher_quality']:
        md += "**Higher quality alternatives:** "
        md += ", ".join(f"`{m}`" for m in metrics['similar_higher_quality'])
        md += "\n\n"

    if not metrics['similar_faster'] and not metrics['similar_higher_quality']:
        md += "*No alternative recommendations available*\n"

    return md


def format_summary_stats(stats: Dict) -> str:
    """
    Format summary statistics as markdown

    Args:
        stats: Dictionary from analyzer.get_summary_stats()

    Returns:
        Formatted markdown string
    """
    md = "# Model Summary\n\n"
    md += f"**Total Models:** {stats['total']}\n\n"
    md += f"- ✓ **Downloaded:** {stats['downloaded']}\n"
    md += f"- ✗ **Missing:** {stats['missing']}\n"

    if stats.get('partial', 0) > 0:
        md += f"- ◐ **Partial:** {stats['partial']}\n"

    if stats.get('unknown', 0) > 0:
        md += f"- ? **Unknown:** {stats['unknown']}\n"

    md += f"\n**Total Downloaded Size:** {stats['total_size_gb']} GB\n"

    return md


def export_model_report(analyzer, output_path: str = None) -> str:
    """
    Export complete model inventory as markdown report

    Args:
        analyzer: ModelAnalyzer instance
        output_path: Optional output file path

    Returns:
        Generated report content as string
    """
    from datetime import datetime

    # Generate filename if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"model_inventory_{timestamp}.md"

    # Build report
    report = "# Wan2GP Model Inventory Report\n\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "---\n\n"

    # Summary stats
    stats = analyzer.get_summary_stats()
    report += format_summary_stats(stats)
    report += "\n---\n\n"

    # Detailed model list
    report += "# All Models\n\n"

    all_models = analyzer.get_all_models()
    all_models_sorted = sorted(all_models, key=lambda m: analyzer.get_model_name(m))

    for model_type in all_models_sorted:
        status = analyzer.get_model_status(model_type)
        metrics = PerformanceDatabase.get_metrics(model_type)
        model_name = analyzer.get_model_name(model_type)
        size_gb = analyzer.get_model_size_gb(model_type)

        icon = get_status_icon(status)
        report += f"## {icon} {model_name} (`{model_type}`)\n\n"
        report += f"- **Status:** {status.capitalize()}\n"
        report += f"- **Speed:** {metrics['speed_tier'].capitalize()}\n"
        report += f"- **VRAM:** {metrics['vram_min_gb']}-{metrics['vram_recommended_gb']} GB\n"
        report += f"- **Quality:** {metrics['quality_tier'].capitalize()}\n"
        report += f"- **Size:** {size_gb} GB\n"
        report += f"- **Best For:** {metrics['best_for']}\n"

        if status == "downloaded":
            files = analyzer.get_model_files(model_type)
            if files:
                report += "\n**Files:**\n"
                for file_info in files:
                    if file_info["status"] == "downloaded":
                        report += f"- `{file_info['path']}`\n"

        report += "\n"

    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        return f"Report exported to: {output_path}"
    except Exception as e:
        return f"Error exporting report: {str(e)}"


def get_filter_choices() -> Dict[str, List[str]]:
    """Get choices for filter dropdowns"""
    return {
        "status": ["all", "downloaded", "missing", "partial"],
        "speed": ["all", "fast", "medium", "slow"],
        "vram": ["all", "≤12GB", "≤24GB", ">24GB"],
        "sort": ["name", "status", "speed", "vram", "quality"]
    }
