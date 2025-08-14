import sys
import os
import traceback
from pathlib import Path

def main():
    try:
        # Force all output to be visible
        print("=== STREAMLIT APP STARTUP DEBUG ===")
        
        # Set up paths first
        repo_root = Path(__file__).parent.parent
        print(f"Repository root: {repo_root}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Change working directory to repo root
        os.chdir(repo_root)
        sys.path.insert(0, str(repo_root))
        print(f"Changed working directory to: {os.getcwd()}")
        
        # Now import streamlit
        import streamlit as st
        
        st.title("🏦 Loan Default Risk System - Debug Mode")
        st.write("✅ Streamlit imported successfully")
        
        # Check file structure
        st.write("## 📁 File Structure Check")
        st.write(f"**Repository Root:** {repo_root}")
        st.write(f"**Current Working Directory:** {os.getcwd()}")
        
        # List all files in repo root
        if repo_root.exists():
            files = list(repo_root.iterdir())
            st.write("**Files in Repository Root:**")
            for item in files:
                st.write(f"- {item.name} ({'📁' if item.is_dir() else '📄'})")
        
        # Check critical directories
        models_dir = Path("models")
        data_dir = Path("data")
        src_dir = Path("src")
        
        st.write(f"**Models Directory:** {'✅ Exists' if models_dir.exists() else '❌ Missing'}")
        st.write(f"**Data Directory:** {'✅ Exists' if data_dir.exists() else '❌ Missing'}")
        st.write(f"**Src Directory:** {'✅ Exists' if src_dir.exists() else '❌ Missing'}")
        
        if models_dir.exists():
            model_files = list(models_dir.glob("*.joblib"))
            st.write(f"**Model Files Found:** {len(model_files)}")
            for file in model_files:
                file_size = file.stat().st_size / (1024*1024)  # MB
                st.write(f"  - {file.name} ({file_size:.1f} MB)")
        
        # Test basic imports
        st.write("## 📦 Testing Basic Imports")
        try:
            import pandas as pd
            import numpy as np
            import plotly.express as px
            st.success("✅ Basic imports (pandas, numpy, plotly) successful")
        except Exception as e:
            st.error(f"❌ Basic import failed: {str(e)}")
            st.code(traceback.format_exc())
            return
        
        # Test custom module import
        st.write("## 🔧 Testing Custom Module Import")
        try:
            from src.explainability import LoanExplainer, load_test_data
            st.success("✅ Custom module import successful")
        except Exception as e:
            st.error(f"❌ Custom module import failed: {str(e)}")
            st.code(traceback.format_exc())
            
            # Show what's actually in the src directory
            if src_dir.exists():
                st.write("**Files in src directory:**")
                for file in src_dir.glob("*"):
                    st.write(f"  - {file.name}")
            return
        
        # Test model loading (with memory monitoring)
        st.write("## 🤖 Testing Model Loading")
        try:
            if models_dir.exists() and any(models_dir.glob("*.joblib")):
                # Load model with error handling
                explainer = LoanExplainer()
                st.success("✅ Model loaded successfully")
                
                # Test data loading
                data_file = Path("data/processed/model_df.parquet")
                if data_file.exists():
                    X_sample, y_sample = load_test_data(str(data_file))
                    st.success(f"✅ Data loaded successfully: {X_sample.shape}")
                else:
                    st.warning("⚠️ Data file not found, using demo mode")
                    
            else:
                st.warning("⚠️ No model files found - demo mode")
                
        except Exception as e:
            st.error(f"❌ Model/Data loading failed: {str(e)}")
            st.code(traceback.format_exc())
            return
        
        # Memory usage check
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            st.write(f"**Current Memory Usage:** {memory_mb:.1f} MB")
            if memory_mb > 800:  # Streamlit Cloud has ~1GB limit
                st.warning("⚠️ High memory usage detected!")
        except:
            st.write("Memory monitoring not available")
        
        # If we reach here, show success
        st.success("🎉 **All tests passed!** Your app structure is working.")
        st.info("The issue was likely in the full app code. You can now gradually add back functionality.")
        
        # Add a simple working interface
        st.write("## 🧪 Simple Test Interface")
        if st.button("Test Button"):
            st.balloons()
            st.write("Button works! Your deployment is functional.")
            
    except Exception as e:
        # This should catch any remaining issues
        print(f"CRITICAL ERROR: {str(e)}")
        print(traceback.format_exc())
        
        # Try to show error even if streamlit fails
        try:
            import streamlit as st
            st.error("❌ **CRITICAL DEPLOYMENT ERROR**")
            st.error(f"**Error:** {str(e)}")
            st.code(traceback.format_exc())
        except:
            # If streamlit itself fails, print to console
            print("STREAMLIT IMPORT FAILED")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
