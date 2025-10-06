#!/usr/bin/env python3
"""
Script principal para treinar todos os modelos Random Forest para os 4 produtos.

Este script executa o treinamento de todos os modelos em sequência:
1. BOMBOM MORANGUETE 13G 160UN
2. TETA BEL TRADICIONAL 50UN  
3. TOPBEL LEITE CONDENSADO 50UN
4. TOPBEL TRADICIONAL 50UN

Cada modelo usa dados de 2021-2023 para treinamento e 2024-2025 para teste.
"""

import subprocess
import sys
import time
from datetime import datetime

def run_model_script(script_name, product_name):
    """Executa um script de modelo e captura o tempo de execução."""
    print(f"\n{'='*80}")
    print(f"INICIANDO TREINAMENTO: {product_name}")
    print(f"Script: {script_name}")
    print(f"Horário: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Executar o script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              cwd='/workspaces/DemandAI')
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ SUCESSO: {product_name}")
            print(f"⏱️ Tempo de execução: {duration:.2f} segundos ({duration/60:.2f} minutos)")
            print("\n📊 SAÍDA DO MODELO:")
            print("-" * 60)
            print(result.stdout)
        else:
            print(f"\n❌ ERRO: {product_name}")
            print(f"⏱️ Tempo até falha: {duration:.2f} segundos")
            print("\n🚨 ERRO:")
            print("-" * 60)
            print(result.stderr)
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n💥 EXCEÇÃO: {product_name}")
        print(f"⏱️ Tempo até exceção: {duration:.2f} segundos")
        print(f"Erro: {str(e)}")
        return False
    
    return True

def main():
    """Função principal que executa todos os modelos."""
    print("🚀 INICIANDO TREINAMENTO DE TODOS OS MODELOS RANDOM FOREST")
    print(f"⏰ Horário de início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Diretório de trabalho: /workspaces/DemandAI")
    print(f"🎯 Dados de treino: 2021-2023")
    print(f"🔍 Dados de teste: 2024-2025")
    
    # Lista de modelos para executar
    models_to_run = [
        ("bombom_moranguete_rf_model.py", "BOMBOM MORANGUETE 13G 160UN"),
        ("teta_bel_rf_model.py", "TETA BEL TRADICIONAL 50UN"),
        ("topbel_leite_condensado_rf_model.py", "TOPBEL LEITE CONDENSADO 50UN"),
        ("topbel_tradicional_rf_model.py", "TOPBEL TRADICIONAL 50UN")
    ]
    
    successful_models = []
    failed_models = []
    total_start_time = time.time()
    
    # Executar cada modelo
    for i, (script, product) in enumerate(models_to_run, 1):
        print(f"\n📈 MODELO {i}/{len(models_to_run)}")
        
        if run_model_script(script, product):
            successful_models.append(product)
        else:
            failed_models.append(product)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Relatório final
    print(f"\n\n{'='*80}")
    print("📋 RELATÓRIO FINAL DE EXECUÇÃO")
    print(f"{'='*80}")
    print(f"⏰ Tempo total de execução: {total_duration:.2f} segundos ({total_duration/60:.2f} minutos)")
    print(f"✅ Modelos bem-sucedidos: {len(successful_models)}/{len(models_to_run)}")
    print(f"❌ Modelos com falha: {len(failed_models)}/{len(models_to_run)}")
    
    if successful_models:
        print(f"\n🎉 MODELOS TREINADOS COM SUCESSO:")
        for i, product in enumerate(successful_models, 1):
            print(f"  {i}. {product}")
    
    if failed_models:
        print(f"\n🚨 MODELOS COM FALHA:")
        for i, product in enumerate(failed_models, 1):
            print(f"  {i}. {product}")
    
    # Verificar arquivos gerados
    print(f"\n📁 VERIFICANDO ARQUIVOS GERADOS:")
    import os
    models_dir = "/workspaces/DemandAI/models/"
    
    expected_files = [
        "bombom_moranguete_rf_model.pkl",
        "bombom_moranguete_rf_model_info.json", 
        "bombom_moranguete_rf_analysis.png",
        "teta_bel_rf_model.pkl",
        "teta_bel_rf_model_info.json",
        "teta_bel_rf_analysis.png",
        "topbel_leite_condensado_rf_model.pkl",
        "topbel_leite_condensado_rf_model_info.json",
        "topbel_leite_condensado_rf_analysis.png",
        "topbel_tradicional_rf_model.pkl",
        "topbel_tradicional_rf_model_info.json",
        "topbel_tradicional_rf_analysis.png"
    ]
    
    existing_files = []
    missing_files = []
    
    for file in expected_files:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            existing_files.append(f"{file} ({file_size:,} bytes)")
        else:
            missing_files.append(file)
    
    if existing_files:
        print(f"✅ Arquivos criados ({len(existing_files)}):")
        for file_info in existing_files:
            print(f"  - {file_info}")
    
    if missing_files:
        print(f"❌ Arquivos não encontrados ({len(missing_files)}):")
        for file in missing_files:
            print(f"  - {file}")
    
    # Status final
    if len(failed_models) == 0:
        print(f"\n🎊 TODOS OS MODELOS FORAM TREINADOS COM SUCESSO!")
        print(f"🔍 Verifique a pasta /workspaces/DemandAI/models/ para os arquivos gerados.")
        return 0
    else:
        print(f"\n⚠️  ALGUNS MODELOS FALHARAM. Verifique os logs acima para detalhes.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)