#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UTF-8 编码检查脚本
用于检查项目中的所有文本文件是否正确使用UTF-8编码，并检测潜在的乱码问题
"""

import os
import sys
import chardet
from pathlib import Path
import json
import re

# ANSI 转义序列用于彩色输出
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_file_encoding(file_path):
    """检查文件编码"""
    try:
        # 读取文件的原始字节
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        # 检测编码
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']

        # 尝试用检测到的编码解码
        if encoding:
            try:
                text = raw_data.decode(encoding)

                # 检查是否包含中文字符
                has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))

                # 如果包含中文但不是UTF-8编码，需要警告
                if has_chinese and encoding.upper() not in ['UTF-8', 'UTF-8-SIG', 'ASCII']:
                    return {
                        'status': 'warning',
                        'encoding': encoding,
                        'confidence': confidence,
                        'has_chinese': has_chinese,
                        'message': f'包含中文但编码为 {encoding}'
                    }

                # 检查BOM
                if raw_data.startswith(b'\xef\xbb\xbf'):
                    return {
                        'status': 'info',
                        'encoding': 'UTF-8-BOM',
                        'confidence': confidence,
                        'has_chinese': has_chinese,
                        'message': '文件包含BOM标记'
                    }

                return {
                    'status': 'ok',
                    'encoding': encoding,
                    'confidence': confidence,
                    'has_chinese': has_chinese,
                    'message': 'OK'
                }

            except UnicodeDecodeError as e:
                return {
                    'status': 'error',
                    'encoding': encoding,
                    'confidence': confidence,
                    'message': f'解码错误: {str(e)}'
                }
        else:
            return {
                'status': 'error',
                'encoding': None,
                'confidence': 0,
                'message': '无法检测编码'
            }

    except Exception as e:
        return {
            'status': 'error',
            'encoding': None,
            'confidence': 0,
            'message': f'读取文件错误: {str(e)}'
        }

def check_chinese_display(file_path):
    """检查中文显示是否正常"""
    chinese_samples = {
        '实时视频分析': 'Real-time Video Analytics',
        '活跃流数量': 'Active Streams',
        '检测详情': 'Detection Details',
        '等待检测数据': 'Waiting for detections',
        '已连接': 'Connected',
        '未连接': 'Disconnected'
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        found_chinese = []
        for chinese, english in chinese_samples.items():
            if chinese in content:
                found_chinese.append(chinese)

        return found_chinese
    except:
        return []

def scan_project(root_path='.'):
    """扫描项目中的所有相关文件"""
    extensions = ['.html', '.js', '.jsx', '.ts', '.tsx', '.css', '.json', '.py', '.md']
    ignore_dirs = ['node_modules', '.git', '__pycache__', 'venv', 'env', '.idea', 'dist', 'build']

    results = {
        'total': 0,
        'ok': 0,
        'warning': 0,
        'error': 0,
        'files': []
    }

    print(f"{Colors.BOLD}开始扫描项目文件编码...{Colors.END}\n")

    for root, dirs, files in os.walk(root_path):
        # 过滤忽略的目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            # 检查文件扩展名
            if any(file.endswith(ext) for ext in extensions):
                file_path = Path(root) / file
                results['total'] += 1

                # 检查编码
                check_result = check_file_encoding(file_path)
                chinese_found = check_chinese_display(file_path)

                file_info = {
                    'path': str(file_path),
                    'encoding': check_result['encoding'],
                    'status': check_result['status'],
                    'message': check_result['message'],
                    'chinese_found': chinese_found
                }

                results['files'].append(file_info)

                # 统计
                if check_result['status'] == 'ok':
                    results['ok'] += 1
                    status_icon = '✅'
                    color = Colors.GREEN
                elif check_result['status'] == 'warning':
                    results['warning'] += 1
                    status_icon = '⚠️'
                    color = Colors.YELLOW
                else:
                    results['error'] += 1
                    status_icon = '❌'
                    color = Colors.RED

                # 输出结果
                rel_path = os.path.relpath(file_path, root_path)

                if check_result['status'] != 'ok' or chinese_found:
                    if check_result['status'] == 'ok':
                        status_text = '[OK]'
                    elif check_result['status'] == 'warning':
                        status_text = '[WARNING]'
                    else:
                        status_text = '[ERROR]'

                    print(f"{status_text} {color}{rel_path}{Colors.END}")
                    print(f"   编码: {check_result['encoding']} (置信度: {check_result.get('confidence', 0):.1%})")
                    if check_result['message'] != 'OK':
                        print(f"   状态: {check_result['message']}")
                    if chinese_found:
                        print(f"   {Colors.BLUE}发现中文: {', '.join(chinese_found[:5])}{Colors.END}")
                    print()

    return results

def generate_report(results):
    """生成检查报告"""
    print(f"\n{Colors.BOLD}编码检查报告{Colors.END}")
    print("=" * 60)
    print(f"扫描文件总数: {results['total']}")
    print(f"{Colors.GREEN}[OK] 正常: {results['ok']}{Colors.END}")
    print(f"{Colors.YELLOW}[WARNING] 警告: {results['warning']}{Colors.END}")
    print(f"{Colors.RED}[ERROR] 错误: {results['error']}{Colors.END}")
    print("=" * 60)

    # 列出有问题的文件
    if results['warning'] > 0 or results['error'] > 0:
        print(f"\n{Colors.BOLD}需要注意的文件:{Colors.END}")
        for file_info in results['files']:
            if file_info['status'] in ['warning', 'error']:
                print(f"  - {file_info['path']}")
                print(f"    编码: {file_info['encoding']}, 问题: {file_info['message']}")

    # 中文本地化统计
    chinese_files = [f for f in results['files'] if f['chinese_found']]
    if chinese_files:
        print(f"\n{Colors.BOLD}中文本地化文件 ({len(chinese_files)}个):{Colors.END}")
        for file_info in chinese_files[:10]:  # 只显示前10个
            print(f"  - {file_info['path']}")

    # 建议
    print(f"\n{Colors.BOLD}建议:{Colors.END}")
    if results['warning'] > 0:
        print("  - 部分文件编码可能有问题，建议转换为UTF-8")
        print("  - 使用命令: iconv -f GBK -t UTF-8 input.txt > output.txt")
    if results['error'] > 0:
        print("  - 发现编码错误的文件，请手动检查并修复")
    print("  - 确保编辑器设置为UTF-8编码")
    print("  - 使用.editorconfig文件统一团队编码规范")

    # 保存JSON报告
    report_file = 'encoding_check_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存到: {report_file}")

def fix_encoding(file_path, target_encoding='utf-8'):
    """尝试修复文件编码"""
    try:
        # 检测当前编码
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        result = chardet.detect(raw_data)
        current_encoding = result['encoding']

        if current_encoding and current_encoding.upper() != target_encoding.upper():
            # 转换编码
            text = raw_data.decode(current_encoding)

            # 备份原文件
            backup_path = f"{file_path}.backup"
            os.rename(file_path, backup_path)

            # 写入新编码
            with open(file_path, 'w', encoding=target_encoding) as f:
                f.write(text)

            print(f"[OK] 已将 {file_path} 从 {current_encoding} 转换为 {target_encoding}")
            print(f"   备份文件: {backup_path}")
            return True
    except Exception as e:
        print(f"[ERROR] 转换失败: {str(e)}")
        return False

def main():
    """主函数"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 44)
    print("     UTF-8 编码检查工具 v1.0")
    print("     检查中文本地化文件编码问题")
    print("=" * 44)
    print(f"{Colors.END}\n")

    # 检查依赖
    try:
        import chardet
    except ImportError:
        print(f"{Colors.RED}错误: 缺少 chardet 库{Colors.END}")
        print("请运行: pip install chardet")
        sys.exit(1)

    # 运行扫描
    project_root = 'F:/pythonproject/realtime-video-analytics-32streams'
    if os.path.exists(project_root):
        os.chdir(project_root)

    results = scan_project('.')
    generate_report(results)

    # 询问是否自动修复
    if results['warning'] > 0:
        print(f"\n{Colors.YELLOW}是否自动修复编码问题？(y/n): {Colors.END}", end='')
        answer = input().strip().lower()
        if answer == 'y':
            for file_info in results['files']:
                if file_info['status'] == 'warning':
                    fix_encoding(file_info['path'])

if __name__ == '__main__':
    main()