import { Link, useLocation } from "react-router-dom";

export function Navigation() {
  const location = useLocation();
  
  const isActive = (path) => location.pathname === path;
  
  const linkClass = (path) => 
    `px-4 py-2 rounded transition-colors ${
      isActive(path) 
        ? 'bg-blue-500 text-white' 
        : 'text-gray-600 hover:bg-gray-100'
    }`;

  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="text-xl font-bold text-gray-800">
            DemandAI
          </Link>
          
          <div className="flex space-x-4">
            <Link to="/" className={linkClass("/")}>
              Dashboard
            </Link>
            <Link to="/predict" className={linkClass("/predict")}>
              Predição
            </Link>
            <Link to="/history" className={linkClass("/history")}>
              Histórico
            </Link>
            {/* <Link to="/login" className={linkClass("/login")}> */}
              {/* Login */}
            {/* </Link> */}
            {/* <Link to="/register" className={linkClass("/register")}> */}
              {/* Cadastro */}
            {/* </Link> */}
          </div>
        </div>
      </div>
    </nav>
  );
}