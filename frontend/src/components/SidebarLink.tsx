import React, { cloneElement, isValidElement, type ReactNode, type ReactElement } from 'react';

interface SidebarLinkProps {
  href: string;
  label: string;
  icon?: ReactNode;
  active?: boolean;
}

export default function SidebarLink({ href, label, icon, active = false }: SidebarLinkProps) {
  let renderedIcon: ReactNode = icon;

  if (isValidElement(icon)) {
    const el = icon as ReactElement;
    const originalColor = (el.props as any).color;
    const color = active ? '#ffffff' : originalColor ?? '#615D5B';
    try {
      renderedIcon = cloneElement(el, { color } as any);
    } catch {
      renderedIcon = icon;
    }
  }

  return (
    <a href={href}>
      <div className={`menu-item ${active ? 'active' : ''}`}>
        {renderedIcon}
        <p>{label}</p>
      </div>
    </a>
  );
}