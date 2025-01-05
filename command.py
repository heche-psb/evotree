import click
import logging
from rich.logging import RichHandler
from evotree.basicdraw import plottree
from evotree.simulatepbmm import pbmmodeling
from evotree.simulatepbdm import pbdmmodeling

@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--verbosity', '-v', type=click.Choice(['info', 'debug']), default='info', help="Verbosity level, default = info.")
def cli(verbosity):
    """
    evotree - Copyright (C) 2025-2026 Hengchi Chen\n
    Contact: heche@psb.vib-ugent.be
    """
    logging.basicConfig(
        format='%(message)s',
        handlers=[RichHandler()],
        datefmt='%H:%M:%S',
        level=verbosity.upper())
    pass

@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--tree', '-tr', default=None, show_default=True,help='Tree file name')
@click.option('--figsize', '-fs', nargs=2, default=(10, 10), show_default=True,help='Figsize of plot')
@click.option('--polar', '-p', default=None, type=float, show_default=True,help='Polar transformation')
@click.option('--trait','-ta', default=None, multiple=True, show_default=True,help='Trait data file name')
@click.option('--usedtraitcolumns','-utc', default=None, multiple=True, show_default=True,help='Used trait columns')
@click.option('--wgd', '-w', default=None, show_default=True,help='WGD data file name')
@click.option('--output', '-o', default='Plottree.pdf', show_default=True,help='Output file name')
def initree(**kwargs):
    TB,Tree = plottree(**kwargs)

@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--tree', '-tr', default=None, show_default=True,help='Tree file name')
@click.option('--trait', '-ta', default=None, show_default=True,help='Trait data file name')
@click.option('--traitcolname', '-tac', default=None, show_default=True,help='Selected column of Trait data')
@click.option('--output', '-o', default=None, show_default=True,help='output file name')
def simulatepbmm(**kwargs):
    pbmmodeling(**kwargs)

@cli.command(context_settings={'help_option_names': ['-h', '--help']})
def simulatepbdm():
    pbdmmodeling()

if __name__ == "__main__":
    cli()
