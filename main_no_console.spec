# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None


a = Analysis(['main.py'],
             pathex=[],
             binaries=[('lsl.dll', '.')],             
			 datas = [('mainwindow.ui', '.')] + collect_data_files("pyqtgraph",
                           includes=["**/*.ui", "**/*.png", "**/*.svg"]),
             hiddenimports = [
			    'pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5',
                'pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt5',
                'pyqtgraph.imageview.ImageViewTemplate_pyqt5',
				'sklearn.metrics._pairwise_distances_reduction._datasets_pair',
				'termcolor', 'Highlighter'
				] + 
				collect_submodules(
                       "pyqtgraph", filter=lambda name: "Template" in name),
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='main_no_console',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )

