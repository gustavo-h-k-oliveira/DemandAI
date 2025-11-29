import React, { useState } from 'react';
import { CaretLeft, SquaresFour, FileArrowUp, Files, Folder, Moon, Sun, SignOut } from 'phosphor-react';
import SidebarLink from './SidebarLink';
import logo from '../static/zd-logo.png';

type SidebarLinkItem = {
  label: string;
  href: string;
  icon: React.ReactNode;
  active?: boolean;
};

const LINKS: SidebarLinkItem[] = [
  { label: 'Dashboard', href: '/', icon: <SquaresFour size={20} color="#615D5B" />, active: true },
  { label: 'Arquivos', href: '/', icon: <FileArrowUp size={20} color="#615D5B" /> },
  { label: 'Relatórios', href: '/', icon: <Files size={20} color="#615D5B" /> },
  { label: 'Histórico', href: '/', icon: <Folder size={20} color="#615D5B" /> }
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  // Placeholder to control rendering of icon/label between dark/light
  // Not implementing toggle functionality yet — this is static for now.
  const isDarkMode = true;

  function handleToggle() {
    setCollapsed((prev) => !prev);
  }

  return (
    <nav className={`sidebar${collapsed ? ' collapsed' : ''}`} aria-expanded={!collapsed}>
      <div className="logo">
        <img src={logo} alt="DemandAI Logo" />
        <button
          className="toggle-nav"
          type="button"
          aria-label={collapsed ? 'Expandir navegação' : 'Recolher navegação'}
          onClick={handleToggle}
        >
          <CaretLeft size={24} weight="bold" color="#615D5B" />
        </button>
      </div>
      {!collapsed && (
        <>
          <div className="menu">
            <p>Menu</p>
            {LINKS.map((link) => (
              <SidebarLink key={link.label} {...link} />
            ))}
          </div>
          <div className='menu'>
            <p>Geral</p>
            <SidebarLink
              label={isDarkMode ? 'Modo Claro' : 'Modo Escuro'}
              href="/"
              icon={isDarkMode ? <Sun size={20} color="#615D5B" /> : <Moon size={20} color="#615D5B" />}
            />
            <SidebarLink label="Sair" href="/" icon={<SignOut size={20} color="#615D5B" />} />
          </div>
        </>
      )}
    </nav>
  );
}
