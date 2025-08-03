#!/usr/bin/env python3
"""
Unified Dashboard Launcher

Simple launcher script for the unified dashboard with command-line options.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from unified_dashboard.main_dashboard import create_unified_dashboard


def main():
    """Main entry point for dashboard launcher"""
    
    parser = argparse.ArgumentParser(
        description="BTC Advanced Feature Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run dashboard on default port 8050
    python run_dashboard.py
    
    # Run on custom port with debug mode
    python run_dashboard.py --port 8080 --debug
    
    # Run with specific database config
    python run_dashboard.py --db-host localhost --db-port 5432
    
    # Run with external access
    python run_dashboard.py --host 0.0.0.0 --port 8050

Available Pages:
    📊 Overview: http://localhost:8050/
    🎯 RPN Features: http://localhost:8050/rpn
    📦 Binning Features: http://localhost:8050/binning
    👑 Dominance Features: http://localhost:8050/dominance
    📡 Signal Features: http://localhost:8050/signals
    🚀 Advanced Features: http://localhost:8050/advanced
    📈 Statistical Analysis: http://localhost:8050/statistical
    🔗 API Documentation: http://localhost:8050/api
        """
    )
    
    # Server options
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port to run on (default: 8050)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    # Database options
    parser.add_argument('--db-host', type=str,
                       help='Database host')
    parser.add_argument('--db-port', type=int,
                       help='Database port')
    parser.add_argument('--db-name', type=str,
                       help='Database name')
    parser.add_argument('--db-user', type=str,
                       help='Database user')
    parser.add_argument('--db-password', type=str,
                       help='Database password')
    parser.add_argument('--db-config', type=str,
                       help='Path to database config file')
    
    # Logging options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Build database config
    db_config = None
    if any([args.db_host, args.db_port, args.db_name, args.db_user, args.db_password]):
        db_config = {}
        if args.db_host:
            db_config['host'] = args.db_host
        if args.db_port:
            db_config['port'] = args.db_port
        if args.db_name:
            db_config['database'] = args.db_name
        if args.db_user:
            db_config['user'] = args.db_user
        if args.db_password:
            db_config['password'] = args.db_password
    
    elif args.db_config:
        import json
        try:
            with open(args.db_config, 'r') as f:
                db_config = json.load(f)
            logger.info(f"Loaded database config from {args.db_config}")
        except Exception as e:
            logger.error(f"Error loading database config: {e}")
            return 1
    
    # Create and run dashboard
    try:
        logger.info("🚀 Starting BTC Advanced Feature Dashboard")
        logger.info(f"🌐 Dashboard will be available at: http://{args.host}:{args.port}")
        
        if args.debug:
            logger.info("🔧 Debug mode enabled")
        
        dashboard = create_unified_dashboard(
            db_config=db_config,
            port=args.port,
            debug=args.debug
        )
        
        dashboard.run_server(host=args.host)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Dashboard stopped by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Dashboard failed to start: {e}")
        return 1


if __name__ == "__main__":
    exit(main())